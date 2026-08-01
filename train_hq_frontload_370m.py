#!/usr/bin/env python3
"""HQ front-load pretraining on RegMix 10B (OLMo2-370M).

Phases (default):
  A  steps 0–23   (~100M)  ``pretrain/regmix-10b``   — LR warmup on general pretrain
  B  steps 24–47  (~100M)  ``pretrain/hq-frontload-100m`` — HQ front-load
  C  steps 48–2383 (~9.8B) ``pretrain/regmix-10b``   — remaining general pretrain

Control arm (``--arm control``): shuffled regmix for all steps (no Phase B).

Ephemeral runtime: stage validated shards from ``s3://edullm-data/`` into job-local
cache. Durable checkpoints sync to ``s3://edullm-checkpoints/alt-cl/<arm_id>/``
when S3 export is enabled. HQ corpus is not published yet — trainer fails closed
unless ``--hq-paths-file`` / ``--allow-missing-hq`` (smoke) is set.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from alt_cl.checkpoint_ladder import permanent_checkpoint_steps
from alt_cl.data import (
    DEFAULT_HQ_DATASET_ID,
    DEFAULT_REGMIX_DATASET_ID,
    default_data_cache_dir,
    read_paths_file,
    resolve_and_stage_train_tokens,
    write_stage_meta,
)
from alt_cl.phases import (
    GLOBAL_BATCH_TOKENS,
    PHASE_A_STEPS,
    PHASE_B_STEPS,
    TOTAL_STEPS,
    corpus_for_step,
    phase_boundaries,
    phase_for_step,
)
from alt_cl.s3_export import s3_export_enabled, sync_to_s3

try:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
except Exception:  # pragma: no cover
    StateDictOptions = None  # type: ignore
    get_model_state_dict = None  # type: ignore
    get_optimizer_state_dict = None  # type: ignore
    set_model_state_dict = None  # type: ignore
    set_optimizer_state_dict = None  # type: ignore

log = logging.getLogger("train_hq_frontload_370m")

SEQ_LEN = 2048
TOKENIZER_ID = "allenai/dolma2-tokenizer"
EMBEDDING_SIZE = 100_352
MICROBATCH_TOKENS = 65_536
PEAK_LR = 4.0e-4
DEFAULT_SEED = 42
DEFAULT_LENGTH_TOKENS = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
CONFIG_NAME = "OLMo-2-370M-hq-frontload"
CHECKPOINT_BUCKET = "edullm-checkpoints"
S3_ROOT = "alt-cl"


def arm_s3_uri(arm_id: str, *parts: str) -> str:
    arm = str(arm_id).strip().strip("/")
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    base = f"s3://{CHECKPOINT_BUCKET}/{S3_ROOT}/{arm}"
    return f"{base}/{extra}" if extra else base


def _broadcast_rank0_success(ok: bool) -> bool:
    if not is_distributed():
        return ok
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.broadcast(flag, src=0)
    return bool(int(flag.item()))


def _abort_all_ranks(message: str, *, ok: bool) -> None:
    if not _broadcast_rank0_success(ok):
        raise SystemExit(message)


@dataclass
class _Bookkeeping:
    global_step: int
    max_steps: int
    global_batch_size: int
    max_tokens: Optional[int] = None
    global_train_tokens_seen: int = 0
    dp_process_group: Any = None
    device: torch.device = torch.device("cuda")
    last_ce_loss: Optional[float] = None
    ce_loss_window_sum: float = 0.0
    ce_loss_window_n: int = 0

    def record_metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_ce_loss(self, value: Any, *args: Any, **kwargs: Any) -> None:
        try:
            if torch.is_tensor(value):
                v = float(value.detach().float().mean().item())
            else:
                v = float(value)
        except Exception:
            return
        self.last_ce_loss = v
        self.ce_loss_window_sum += v
        self.ce_loss_window_n += 1

    def pop_ce_loss_avg(self) -> Optional[float]:
        if self.ce_loss_window_n <= 0:
            return self.last_ce_loss
        avg = self.ce_loss_window_sum / float(self.ce_loss_window_n)
        self.ce_loss_window_sum = 0.0
        self.ce_loss_window_n = 0
        return avg


class MemmapTokenDataset(Dataset):
    """Contiguous SEQ_LEN chunks over one or more uint32 token memmaps."""

    def __init__(
        self,
        paths: List[str],
        chunk_size: int = SEQ_LEN,
        dtype: Any = np.uint32,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=dtype)
            n = (len(mm) - 1) // self.chunk_size
            if n <= 0:
                continue
            self._mmaps.append(mm)
            total += n
            self._cum_chunks.append(total)
        if total <= 0:
            raise SystemExit(f"No usable chunks in {len(paths)} paths")
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._total
        prev = 0
        for mm, cum in zip(self._mmaps, self._cum_chunks):
            if idx < cum:
                local = idx - prev
                start = local * self.chunk_size
                arr = np.asarray(
                    mm[start : start + self.chunk_size + 1], dtype=np.int64
                )
                return torch.from_numpy(arr[:-1].copy())
            prev = cum
        raise IndexError(idx)


def collate_input_ids(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"input_ids": torch.stack(batch, dim=0)}


class InfiniteBatchStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.pin_memory = torch.cuda.is_available()
        self._epoch = 0
        self._loader: Optional[DataLoader] = None
        self._it: Optional[Iterator] = None

    def _make_loader(self) -> DataLoader:
        if self.world_size > 1:
            sampler: Any = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed,
                drop_last=True,
            )
            sampler.set_epoch(self._epoch)
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * 1_000_003)
            sampler = RandomSampler(self.dataset, replacement=False, generator=g)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_input_ids,
        )

    def next_batch(self) -> Dict[str, torch.Tensor]:
        if self._it is None:
            self._loader = self._make_loader()
            self._it = iter(self._loader)
        while True:
            try:
                return next(self._it)
            except StopIteration:
                self._epoch += 1
                self._loader = self._make_loader()
                self._it = iter(self._loader)


def next_rank_input_ids(
    stream: InfiniteBatchStream, n_seqs: int, device: torch.device
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    got = 0
    while got < n_seqs:
        x = stream.next_batch()["input_ids"]
        chunks.append(x)
        got += x.size(0)
    return torch.cat(chunks, dim=0)[:n_seqs].to(device, non_blocking=True)


class PhasedBatchStream:
    """Switches between regmix / HQ infinite streams by global step."""

    def __init__(
        self,
        regmix: InfiniteBatchStream,
        hq: Optional[InfiniteBatchStream],
        *,
        phases,
        control: bool,
        seqs_per_rank: int,
        device: torch.device,
    ) -> None:
        self.regmix = regmix
        self.hq = hq
        self.phases = phases
        self.control = control
        self.seqs_per_rank = int(seqs_per_rank)
        self.device = device

    def next_input_ids(self, step: int) -> Tuple[torch.Tensor, str]:
        corpus = corpus_for_step(step, self.phases, control=self.control)
        if corpus == "hq":
            if self.hq is None:
                raise SystemExit(
                    f"step {step} needs HQ corpus but no HQ stream was staged "
                    f"(publish {DEFAULT_HQ_DATASET_ID} or pass --hq-paths-file)"
                )
            stream = self.hq
        else:
            stream = self.regmix
        return next_rank_input_ids(stream, self.seqs_per_rank, self.device), corpus


def resolve_attn_backend() -> AttentionBackendName:
    prefer = os.environ.get("OLMO_ATTN_BACKEND", "torch").strip().lower()
    if prefer in ("torch", "sdpa", "eager"):
        return AttentionBackendName.torch
    if prefer in ("flash_2", "flash", "flash2", "auto"):
        try:
            import flash_attn  # noqa: F401

            backend = AttentionBackendName.flash_2
            backend.get_class().assert_supported()
            return backend
        except Exception as e:
            log.warning("flash_attn unavailable (%s); using torch", e)
            return AttentionBackendName.torch
    try:
        return AttentionBackendName(prefer)
    except Exception:
        return AttentionBackendName.torch


def build_olmo2_config(*, fused_ce: bool) -> TransformerConfig:
    vocab_size = TokenizerConfig.dolma2().padded_vocab_size()
    if vocab_size != EMBEDDING_SIZE:
        raise SystemExit(
            f"dolma2 padded vocab {vocab_size} != expected EMBEDDING_SIZE {EMBEDDING_SIZE}"
        )
    cfg = TransformerConfig.olmo2_370M(
        vocab_size=vocab_size,
        attn_backend=resolve_attn_backend(),
    )
    if fused_ce:
        try:
            cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
        except Exception:
            pass
    return cfg


def try_enable_fused_ce() -> bool:
    try:
        import liger_kernel  # noqa: F401

        return True
    except Exception:
        return False


def build_train_module(
    *,
    lr: float,
    lr_warmup_steps: int,
    alpha_f: float,
    compile_model: bool,
    rank_microbatch_tokens: int,
) -> TransformerTrainModule:
    fused = try_enable_fused_ce()
    model_cfg = build_olmo2_config(fused_ce=fused)
    try:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps, alpha_f=alpha_f)
    except TypeError:
        scheduler = CosWithWarmup(warmup_steps=lr_warmup_steps)
        if hasattr(scheduler, "alpha_f"):
            scheduler.alpha_f = alpha_f

    tm_cfg = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQ_LEN,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=scheduler,
    )
    model = model_cfg.build(init_device="cuda")
    return tm_cfg.build(model)


def _cpu_plain_tensor(t: Any) -> torch.Tensor:
    if torch.is_tensor(t) and type(t).__name__ == "Tensor":
        return t.detach().cpu()
    full = getattr(t, "full_tensor", None)
    if callable(full):
        try:
            return full().detach().cpu()
        except Exception:
            pass
    local = getattr(t, "to_local", None)
    if callable(local):
        try:
            return local().detach().cpu()
        except Exception:
            pass
    if torch.is_tensor(t):
        return t.detach().cpu()
    raise TypeError(f"cannot convert {type(t)} to CPU tensor")


def _plainify_state_tree(obj: Any) -> Any:
    if torch.is_tensor(obj) or type(obj).__name__ == "DTensor":
        return _cpu_plain_tensor(obj)
    if isinstance(obj, dict):
        return {k: _plainify_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_plainify_state_tree(v) for v in obj]
        return type(obj)(seq) if not isinstance(obj, list) else seq
    return obj


def gather_train_module_state_dict(train_module: TransformerTrainModule) -> dict[str, Any]:
    if get_model_state_dict is None or StateDictOptions is None:
        return _plainify_state_tree(train_module.state_dict_to_save())
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(train_module.model, options=opts)
    optim_sd: Any = None
    if get_optimizer_state_dict is not None:
        try:
            optim_sd = get_optimizer_state_dict(
                train_module.model, train_module.optim, options=opts
            )
        except Exception as e:
            log.warning("full optimizer state gather failed (%s); saving model only", e)
    return {
        "model": _plainify_state_tree(model_sd),
        "optim": _plainify_state_tree(optim_sd) if optim_sd is not None else None,
    }


def save_checkpoint(
    path: Path,
    step: int,
    train_module: TransformerTrainModule,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    train_module_sd = gather_train_module_state_dict(train_module)
    ok = True
    err = "durable S3 export / checkpoint save failed"
    if get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=True)
            state = {
                "step": step,
                "train_module": train_module_sd,
                "args": vars(args),
                "meta": meta,
                "architecture": "olmo_core.TransformerConfig.olmo2_370M",
                "config_name": CONFIG_NAME,
                "method": "hq_frontload" if args.arm != "control" else "control",
                "arm": args.arm_id,
                "run_id": args.name,
                "checkpoint_format": "full_state_dict_v1",
            }
            tmp = path / "state.pt.tmp"
            torch.save(state, tmp)
            tmp.replace(path / "state.pt")
            (path / "step.txt").write_text(str(step) + "\n")
            log.info("Saved checkpoint → %s (step=%s)", path, step)
            if s3_export_enabled(bool(args.s3_export)):
                sync_to_s3(
                    path,
                    arm_s3_uri(args.arm_id, "checkpoints", path.name),
                    enabled=True,
                    fail_closed=True,
                )
                sync_to_s3(
                    Path(args.progress_dir),
                    arm_s3_uri(args.arm_id, "progress"),
                    enabled=True,
                    fail_closed=True,
                )
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"durable S3 export / checkpoint save failed: {exc}"
            log.error("%s", err)
    _abort_all_ranks(err, ok=ok)


def load_checkpoint(path: Path, train_module: TransformerTrainModule) -> int:
    ckpt = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    tm_sd = ckpt["train_module"]
    fmt = ckpt.get("checkpoint_format")
    if (
        fmt == "full_state_dict_v1"
        and isinstance(tm_sd, dict)
        and "model" in tm_sd
        and set_model_state_dict is not None
        and StateDictOptions is not None
    ):
        opts = StateDictOptions(full_state_dict=True, strict=True)
        set_model_state_dict(train_module.model, tm_sd["model"], options=opts)
        if tm_sd.get("optim") is not None and set_optimizer_state_dict is not None:
            try:
                set_optimizer_state_dict(
                    train_module.model,
                    train_module.optim,
                    tm_sd["optim"],
                    options=opts,
                )
            except Exception as e:
                log.warning("optimizer restore failed (%s)", e)
    else:
        train_module.load_state_dict(tm_sd)
    return int(ckpt["step"])


def find_latest_checkpoint(save_folder: Path) -> Optional[Path]:
    if not save_folder.is_dir():
        return None
    cands = [
        p
        for p in save_folder.iterdir()
        if p.is_dir() and p.name.startswith("step") and (p / "state.pt").is_file()
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: int(p.name.replace("step", "").split("-")[0]))


def _stage_corpus(
    *,
    label: str,
    dataset_id: str,
    version: Optional[str],
    paths_file: Optional[str],
    cache_dir: Path,
    allow_missing: bool,
) -> Tuple[Optional[List[str]], Optional[str], Any]:
    if paths_file:
        paths = read_paths_file(Path(paths_file))
        return paths, "local-paths-file", np.uint32
    try:
        paths, ver, dtype = resolve_and_stage_train_tokens(
            dataset_id=dataset_id,
            version=version,
            cache_dir=cache_dir / label,
        )
        write_stage_meta(
            cache_dir / f"_stage_{label}.json",
            dataset_id=dataset_id,
            version=ver,
            paths=paths,
            dtype=dtype,
        )
        return paths, ver, dtype
    except SystemExit:
        if allow_missing:
            log.warning(
                "allow-missing: skipping %s (%s) — not staged", label, dataset_id
            )
            return None, None, None
        raise


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=None)
    ap.add_argument("--arm-id", default=None)
    ap.add_argument(
        "--arm",
        choices=("hq-frontload", "control"),
        default="hq-frontload",
        help="hq-frontload = phases A/B/C; control = regmix only",
    )
    ap.add_argument("--regmix-dataset-id", default=DEFAULT_REGMIX_DATASET_ID)
    ap.add_argument("--regmix-version", default=None)
    ap.add_argument("--hq-dataset-id", default=DEFAULT_HQ_DATASET_ID)
    ap.add_argument("--hq-version", default=None)
    ap.add_argument("--regmix-paths-file", default=None)
    ap.add_argument("--hq-paths-file", default=None)
    ap.add_argument(
        "--allow-missing-hq",
        action="store_true",
        help="Smoke only: skip HQ staging (Phase B will fail if reached)",
    )
    ap.add_argument("--data-cache-dir", default=None)
    ap.add_argument("--save-folder", required=True)
    ap.add_argument("--progress-dir", required=True)
    ap.add_argument("--length-tokens", type=int, default=DEFAULT_LENGTH_TOKENS)
    ap.add_argument("--device-batch-size", type=int, default=MICROBATCH_TOKENS)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--load-path", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--lr-warmup-steps", type=int, default=PHASE_A_STEPS)
    ap.add_argument("--lr-alpha-f", type=float, default=1.0)
    ap.add_argument("--phase-a-steps", type=int, default=PHASE_A_STEPS)
    ap.add_argument("--phase-b-steps", type=int, default=PHASE_B_STEPS)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--s3-export", dest="s3_export", action="store_true", default=True)
    ap.add_argument("--no-s3-export", dest="s3_export", action="store_false")
    ap.add_argument("--allow-local-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.arm_id is None:
        args.arm_id = "control" if args.arm == "control" else "hq-frontload"
    if args.name is None:
        args.name = args.arm_id
    if not args.s3_export and not args.allow_local_only:
        raise SystemExit(
            "S3 export disabled without --allow-local-only "
            "(local smoke: --no-s3-export --allow-local-only)"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    prepare_training_environment()
    try:
        _run(args)
    finally:
        teardown_training_environment()


def _run(args: argparse.Namespace) -> None:
    rank = get_rank()
    world_size = get_world_size() if is_distributed() else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed + rank)

    rank_micro_tokens = int(args.device_batch_size)
    if GLOBAL_BATCH_TOKENS % (world_size * rank_micro_tokens) != 0:
        raise SystemExit(
            f"global_batch_tokens {GLOBAL_BATCH_TOKENS} not divisible by "
            f"world_size({world_size}) * device_batch_size({rank_micro_tokens})"
        )
    seqs_per_rank = GLOBAL_BATCH_TOKENS // (SEQ_LEN * world_size)
    tokens_per_step = GLOBAL_BATCH_TOKENS
    total_steps = int(args.length_tokens) // tokens_per_step
    if total_steps <= 0:
        raise SystemExit("length-tokens too small for one step")

    phases = phase_boundaries(
        phase_a_steps=int(args.phase_a_steps),
        phase_b_steps=int(args.phase_b_steps),
        total_steps=total_steps,
    )
    control = args.arm == "control"
    ladder = permanent_checkpoint_steps(total_steps)
    ladder_set = set(ladder)

    save_folder = Path(args.save_folder)
    progress_dir = Path(args.progress_dir)
    metrics_dir = progress_dir / "metrics"
    if rank == 0:
        save_folder.mkdir(parents=True, exist_ok=True)
        progress_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.data_cache_dir) if args.data_cache_dir else default_data_cache_dir()

    # Rank 0 stages; all ranks read the written meta.
    if rank == 0:
        regmix_paths, regmix_ver, regmix_dtype = _stage_corpus(
            label="regmix",
            dataset_id=args.regmix_dataset_id,
            version=args.regmix_version,
            paths_file=args.regmix_paths_file,
            cache_dir=cache_dir,
            allow_missing=False,
        )
        assert regmix_paths is not None
        hq_paths, hq_ver, hq_dtype = (None, None, None)
        if not control:
            hq_paths, hq_ver, hq_dtype = _stage_corpus(
                label="hq",
                dataset_id=args.hq_dataset_id,
                version=args.hq_version,
                paths_file=args.hq_paths_file,
                cache_dir=cache_dir,
                allow_missing=bool(args.allow_missing_hq),
            )
        stage = {
            "regmix": {
                "dataset_id": args.regmix_dataset_id,
                "version": regmix_ver,
                "paths": regmix_paths,
                "dtype": str(regmix_dtype),
            },
            "hq": {
                "dataset_id": args.hq_dataset_id,
                "version": hq_ver,
                "paths": hq_paths,
                "dtype": str(hq_dtype) if hq_dtype is not None else None,
            },
            "phases": [
                {
                    "name": p.name,
                    "start": p.start_step,
                    "end": p.end_step,
                    "corpus": p.corpus,
                    "tokens": p.n_tokens,
                }
                for p in phases
            ],
            "control": control,
        }
        (cache_dir / "_stage_alt_cl.json").write_text(
            json.dumps(stage, indent=2) + "\n", encoding="utf-8"
        )
        (progress_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "arm_id": args.arm_id,
                    "arm": args.arm,
                    "total_steps": total_steps,
                    "phases": stage["phases"],
                    "regmix": stage["regmix"]["dataset_id"],
                    "hq": stage["hq"]["dataset_id"],
                    "architecture": CONFIG_NAME,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if is_distributed():
        dist.barrier()

    stage = json.loads((cache_dir / "_stage_alt_cl.json").read_text(encoding="utf-8"))
    regmix_ds = MemmapTokenDataset(
        list(stage["regmix"]["paths"]),
        dtype=np.dtype(stage["regmix"]["dtype"] or "uint32"),
    )
    hq_ds: Optional[MemmapTokenDataset] = None
    if stage["hq"]["paths"]:
        hq_ds = MemmapTokenDataset(
            list(stage["hq"]["paths"]),
            dtype=np.dtype(stage["hq"]["dtype"] or "uint32"),
        )

    mbs = max(1, rank_micro_tokens // SEQ_LEN)
    regmix_stream = InfiniteBatchStream(
        regmix_ds, mbs, args.num_workers, args.seed, rank, world_size
    )
    hq_stream = (
        InfiniteBatchStream(
            hq_ds, mbs, args.num_workers, args.seed + 17, rank, world_size
        )
        if hq_ds is not None
        else None
    )
    phased = PhasedBatchStream(
        regmix_stream,
        hq_stream,
        phases=phases,
        control=control,
        seqs_per_rank=seqs_per_rank,
        device=device,
    )

    train_module = build_train_module(
        lr=PEAK_LR,
        lr_warmup_steps=int(args.lr_warmup_steps),
        alpha_f=float(args.lr_alpha_f),
        compile_model=bool(args.compile),
        rank_microbatch_tokens=rank_micro_tokens,
    )
    books = _Bookkeeping(
        global_step=0,
        max_steps=total_steps,
        global_batch_size=GLOBAL_BATCH_TOKENS,
        device=device,
    )
    train_module._attach_trainer(books)  # type: ignore[arg-type]

    meta = {
        "arm": args.arm,
        "arm_id": args.arm_id,
        "phases": stage["phases"],
        "regmix_dataset": args.regmix_dataset_id,
        "hq_dataset": args.hq_dataset_id,
    }

    start_step = 0
    if args.load_path:
        start_step = load_checkpoint(Path(args.load_path), train_module)
    elif args.fresh:
        pass
    else:
        leftover = find_latest_checkpoint(save_folder)
        if leftover is not None:
            raise SystemExit(
                f"found local checkpoint {leftover}; pass --load-path or --fresh"
            )

    if is_distributed():
        dist.barrier()

    if start_step == 0 and 0 in ladder_set:
        save_checkpoint(save_folder / "step0", 0, train_module, args, meta)
        if is_distributed():
            dist.barrier()

    t0 = time.time()
    loss_path = metrics_dir / "train_loss.jsonl"
    for step in range(start_step, total_steps):
        books.global_step = step
        books.global_train_tokens_seen = step * tokens_per_step
        input_ids, corpus = phased.next_input_ids(step)
        phase = phase_for_step(step, phases)
        batch = {"input_ids": input_ids}

        train_module.zero_grads()
        train_module.train_batch(batch)
        train_module.optim_step()

        global_step = step + 1
        if global_step % args.log_interval == 0 or global_step == 1:
            if rank == 0:
                loss_avg = books.pop_ce_loss_avg()
                try:
                    lr_now = float(train_module.optim.param_groups[0]["lr"])
                except Exception:
                    lr_now = float(PEAK_LR)
                elapsed = time.time() - t0
                tok_s = (global_step - start_step) * tokens_per_step / max(elapsed, 1e-6)
                log.info(
                    "step=%d/%d phase=%s corpus=%s loss=%s tok/s=%.0f lr=%.2e",
                    global_step,
                    total_steps,
                    phase.name,
                    corpus,
                    f"{loss_avg:.4f}" if loss_avg is not None else "n/a",
                    tok_s,
                    lr_now,
                )
                row = {
                    "step": global_step,
                    "phase": phase.name,
                    "corpus": corpus,
                    "loss": loss_avg,
                    "lr": lr_now,
                    "tok_per_s": tok_s,
                }
                with loss_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                (progress_dir / "progress.json").write_text(
                    json.dumps(
                        {
                            **row,
                            "total_steps": total_steps,
                            "pct": round(100.0 * global_step / total_steps, 4),
                        }
                    )
                    + "\n"
                )

        if global_step in ladder_set:
            if is_distributed():
                dist.barrier()
            save_checkpoint(
                save_folder / f"step{global_step}",
                global_step,
                train_module,
                args,
                meta,
            )
            if is_distributed():
                dist.barrier()

    if rank == 0:
        if s3_export_enabled(bool(args.s3_export)):
            sync_to_s3(
                save_folder,
                arm_s3_uri(args.arm_id, "checkpoints"),
                enabled=True,
                fail_closed=True,
            )
            sync_to_s3(
                progress_dir,
                arm_s3_uri(args.arm_id, "progress"),
                enabled=True,
                fail_closed=True,
            )
        log.info(
            "Training complete step=%d arm=%s durable=%s",
            total_steps,
            args.arm_id,
            arm_s3_uri(args.arm_id) if args.s3_export else "s3-off",
        )


if __name__ == "__main__":
    main()
