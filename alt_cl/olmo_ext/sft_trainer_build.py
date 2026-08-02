"""Assemble an OLMo-core Trainer for math SFT (load pretrain ckpt + masked CE)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from alt_cl.checkpoint_ladder import permanent_checkpoint_steps
from alt_cl.olmo_ext.callbacks import PhaseMetricsCallback
from alt_cl.olmo_ext.sft_loader import PackedSFTDataLoader
from alt_cl.sft_data import ConversationSFTDataset, IGNORE_INDEX

log = logging.getLogger("alt_cl.olmo_ext.sft_trainer_build")

SEQ_LEN_DEFAULT = 2048
EMBEDDING_SIZE = 100_352


def _resolve_attn_backend():
    import os

    from olmo_core.nn.attention import AttentionBackendName

    prefer = os.environ.get("OLMO_ATTN_BACKEND", "torch").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    if prefer in ("flash_2", "flash", "flash2", "auto"):
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            backend.get_class().assert_supported()
            return backend
        except Exception as e:  # noqa: BLE001
            log.warning("flash_attn unavailable (%s); using torch", e)
            return AttentionBackendName.torch
    try:
        return AttentionBackendName(prefer)
    except Exception:
        return AttentionBackendName.torch


def build_sft_trainer(
    *,
    save_folder: str | Path,
    metrics_dir: str | Path,
    work_dir: str | Path,
    sft_jsonl_gz: str | Path,
    load_path: str | Path,
    max_tokens: int,
    global_batch_size: int,
    sequence_length: int = SEQ_LEN_DEFAULT,
    lr: float = 1.0e-4,
    warmup_steps: int = 50,
    lr_alpha_f: float = 0.1,
    rank_microbatch_size: int = 16_384,
    seed: int = 42_069_666,
    init_seed: Optional[int] = None,
    model_arch: str = "olmo2_370M",
    compile_model: bool = True,
    resume: bool = False,
    checkpoint_every_steps: int = 25,
    checkpoint_keep_last: Any = None,
    ephemeral_checkpoint_every_steps: Optional[int] = None,
    pre_train_checkpoint: bool = True,
    save_async: bool = True,
    log_interval: int = 5,
    max_examples: Optional[int] = None,
    tokenizer_id: str = "allenai/dolma2-tokenizer",
) -> Any:
    """Build Trainer that loads ``load_path`` weights and fits masked SFT."""
    try:
        from olmo_core.config import DType
        from olmo_core.data import TokenizerConfig
        from olmo_core.data.collator import DataCollator
        from olmo_core.distributed.parallel import DataParallelType
        from olmo_core.distributed.utils import get_world_size
        from olmo_core.nn.lm_head import LMLossImplementation
        from olmo_core.nn.transformer import TransformerConfig
        from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
        from olmo_core.train import Duration, LoadStrategy, TrainerConfig
        from olmo_core.train.callbacks import CheckpointerCallback
        from olmo_core.train.train_module import (
            TransformerDataParallelConfig,
            TransformerTrainModuleConfig,
        )
    except ImportError as e:
        raise SystemExit(
            "olmo_core not installed. pip install -e /path/to/edu-llm/OLMo-core\n"
            f"Original error: {e}"
        ) from e

    init_seed = int(seed if init_seed is None else init_seed)
    seq_len = int(sequence_length)
    gbs = int(global_batch_size)
    total_steps = int(max_tokens) // gbs
    if total_steps <= 0:
        raise SystemExit("max_tokens too small for one step")

    load_path_p = Path(load_path)
    if not str(load_path).startswith("s3://") and not load_path_p.exists():
        raise SystemExit(f"SFT load_path not found: {load_path}")

    log.info("tokenizing SFT mix from %s ...", sft_jsonl_gz)
    ds = ConversationSFTDataset(
        sft_jsonl_gz,
        tokenizer_id=tokenizer_id,
        max_length=seq_len,
        max_examples=max_examples,
    )
    input_ids, labels = ds.packed_tensors(seq_len=seq_len)
    if input_ids.shape[0] == 0:
        raise SystemExit("SFT packing produced zero sequences")
    log.info(
        "packed %s sequences × %s tokens (supervised frac≈%.3f)",
        f"{input_ids.shape[0]:,}",
        seq_len,
        float((labels != IGNORE_INDEX).sum().item()) / float(labels.numel()),
    )

    tokenizer = TokenizerConfig.dolma2()
    vocab_size = tokenizer.padded_vocab_size()
    if vocab_size != EMBEDDING_SIZE:
        raise SystemExit(
            f"dolma2 padded vocab {vocab_size} != expected EMBEDDING_SIZE {EMBEDDING_SIZE}"
        )

    model_builder = getattr(TransformerConfig, model_arch, None)
    if model_builder is None or not callable(model_builder):
        raise SystemExit(f"TransformerConfig has no builder {model_arch!r}")
    model_cfg = model_builder(
        vocab_size=vocab_size,
        attn_backend=_resolve_attn_backend(),
        init_seed=init_seed,
    )
    try:
        import liger_kernel  # noqa: F401

        model_cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    except Exception:
        pass

    try:
        scheduler = CosWithWarmup(
            warmup_steps=int(warmup_steps), alpha_f=float(lr_alpha_f)
        )
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=int(warmup_steps))
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = float(lr_alpha_f)

    tm_kwargs = dict(
        rank_microbatch_size=int(rank_microbatch_size),
        max_sequence_length=seq_len,
        optim=SkipStepAdamWConfig(
            lr=float(lr),
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=bool(compile_model),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    import inspect

    if "label_ignore_index" in inspect.signature(TransformerTrainModuleConfig).parameters:
        tm_kwargs["label_ignore_index"] = IGNORE_INDEX
    train_module_cfg = TransformerTrainModuleConfig(**tm_kwargs)

    ladder = permanent_checkpoint_steps(
        total_steps, interval=int(checkpoint_every_steps)
    )
    fixed: List[int] = [s for s in ladder if 0 < s < total_steps]

    ckpt_kwargs: Dict[str, Any] = {
        "save_interval": int(checkpoint_every_steps),
        "max_checkpoints": checkpoint_keep_last,
        "fixed_steps": fixed,
        "pre_train_checkpoint": bool(pre_train_checkpoint),
        "save_async": bool(save_async),
        "ephemeral_save_interval": (
            int(ephemeral_checkpoint_every_steps)
            if ephemeral_checkpoint_every_steps is not None
            else None
        ),
    }

    from alt_cl.platform_env import ensure_local_dir, is_remote_uri

    save_folder_str = str(save_folder)
    if not is_remote_uri(save_folder_str):
        ensure_local_dir(save_folder_str)
    metrics_dir_p = ensure_local_dir(metrics_dir)
    work_dir_p = ensure_local_dir(work_dir)

    # Load pretrain weights; fresh optimizer for SFT unless --resume.
    trainer_cfg = (
        TrainerConfig(
            save_folder=save_folder_str,
            load_path=str(load_path),
            load_strategy=LoadStrategy.always
            if not resume
            else LoadStrategy.if_available,
            load_trainer_state=bool(resume),
            load_optim_state=bool(resume),
            max_duration=Duration.tokens(int(max_tokens)),
        )
        .with_callback("checkpointer", CheckpointerCallback(**ckpt_kwargs))
        .with_callback(
            "phase_metrics",
            PhaseMetricsCallback(
                metrics_dir=str(metrics_dir_p),
                log_interval=int(log_interval),
            ),
        )
    )

    model = model_cfg.build(init_device="meta")
    train_module = train_module_cfg.build(model)

    try:
        dp_pg = train_module.dp_process_group
        from olmo_core.distributed.utils import get_rank as _gr
        from olmo_core.distributed.utils import get_world_size as _gws

        world = _gws(dp_pg) if dp_pg is not None else max(1, get_world_size())
        dp_rank = _gr(dp_pg) if dp_pg is not None else 0
    except Exception:
        world, dp_rank = 1, 0

    collator = DataCollator(pad_token_id=int(tokenizer.pad_token_id))
    data_loader = PackedSFTDataLoader(
        input_ids=input_ids,
        labels=labels,
        sequence_length=seq_len,
        global_batch_size=gbs,
        work_dir=work_dir_p,
        seed=int(seed),
        collator=collator,
        dp_world_size=world,
        dp_rank=int(dp_rank),
        total_steps=total_steps,
    )

    return trainer_cfg.build(train_module=train_module, data_loader=data_loader)
