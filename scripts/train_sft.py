#!/usr/bin/env python3
"""OLMo-core SFT entry for alt-cl math finetune (~60M masked tokens).

Loads a pretrain arm checkpoint, trains on ``artifacts/math-sft-60m/train.jsonl.gz``
(or an ``s3://`` URI) with Dolma2 chat template + assistant-only loss.

  # Dry-run plan:
  python -m scripts.train_sft --config configs/math_sft_60m.yaml \\
      --load-path /path/to/pretrain/step_final

  # Launch (platform):
  bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
    .edullm/train_sft.py "$EDULLM_RUN_ID" --config configs/math_sft_60m.yaml \\
    --load-path "$PRETRAIN_CKPT" --save-folder "$EDULLM_CHECKPOINT_DIR"'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alt_cl.contract import (  # noqa: E402
    load_config,
    resolve_run_scratch,
    resolve_save_folder,
    verify_olmo_revision,
)
from alt_cl.platform_env import (  # noqa: E402
    ensure_local_dir,
    is_remote_uri,
    platform_run_id,
)
from scripts.train_olmo import _olmo_core_importable, pin_cuda_visible_devices  # noqa: E402

log = logging.getLogger("scripts.train_sft")

DEFAULT_SFT_PATH = ROOT / "artifacts" / "math-sft-60m" / "train.jsonl.gz"


def validate_sft_config(cfg: Dict[str, Any]) -> None:
    run_id = str(cfg.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    if cfg.get("seed") is None:
        raise ValueError("seed is required")
    data = cfg.get("data") or {}
    if not data.get("tokenizer"):
        raise ValueError("data.tokenizer is required")
    if int(data.get("sequence_length") or 0) <= 0:
        raise ValueError("data.sequence_length must be > 0")
    model = cfg.get("model") or {}
    if not model.get("arch"):
        raise ValueError("model.arch is required")
    olmo = cfg.get("olmo_core") or {}
    if not str(olmo.get("revision") or "").strip():
        raise ValueError("olmo_core.revision is required")
    train = cfg.get("train") or {}
    max_tokens = int(train.get("max_tokens") or 0)
    gbs = int(train.get("global_batch_size") or 0)
    if max_tokens <= 0 or gbs <= 0:
        raise ValueError("train.max_tokens and train.global_batch_size must be > 0")
    if max_tokens % gbs != 0:
        raise ValueError(
            f"train.max_tokens ({max_tokens}) must be an exact multiple of "
            f"global_batch_size ({gbs})"
        )
    seq = int(data["sequence_length"])
    if gbs % seq != 0:
        raise ValueError(
            f"global_batch_size {gbs} must be divisible by sequence_length {seq}"
        )


def _stage_sft_jsonl(path: str | Path, *, cache_dir: Path) -> Path:
    """Return a local path to train.jsonl.gz (download if ``s3://``)."""
    raw = str(path).strip()
    if not is_remote_uri(raw):
        p = Path(raw)
        if not p.is_file():
            raise SystemExit(
                f"SFT mix not found: {p}\n"
                "Build with: python scripts/build_math_sft_60m.py\n"
                "Or pass --sft-path s3://…/train.jsonl.gz"
            )
        return p
    import boto3

    rest = raw[len("s3://") :]
    bucket, _, key = rest.partition("/")
    dest = cache_dir / "sft" / Path(key).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        log.info("staging SFT mix %s → %s", raw, dest)
        boto3.client("s3", region_name="us-east-1").download_file(
            bucket, key, str(dest)
        )
    return dest


def build_plan(
    cfg: Dict[str, Any],
    *,
    save_folder: str,
    scratch: Dict[str, str],
    load_path: str,
    sft_path: str,
) -> Dict[str, Any]:
    train = cfg.get("train") or {}
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    olmo = cfg.get("olmo_core") or {}
    max_tokens = int(train["max_tokens"])
    gbs = int(train["global_batch_size"])
    ephemeral = train.get("ephemeral_checkpoint_every_steps", None)
    return {
        "run_id": cfg["run_id"],
        "seed": int(cfg["seed"]),
        "init_seed": int(model.get("init_seed", cfg["seed"])),
        "model_name": model.get("name"),
        "model_arch": str(model["arch"]),
        "olmo_revision": str(olmo.get("revision") or ""),
        "tokenizer": str(data["tokenizer"]),
        "sequence_length": int(data["sequence_length"]),
        "max_tokens": max_tokens,
        "global_batch_size": gbs,
        "total_steps": max_tokens // gbs,
        "lr": float(train["lr"]),
        "warmup_steps": int(train.get("warmup_steps", 50)),
        "lr_alpha_f": float(train.get("lr_alpha_f", 0.1)),
        "rank_microbatch_size": int(train.get("rank_microbatch_size", 16384)),
        "compile_model": bool(train.get("compile_model", True)),
        "checkpoint_every_steps": int(train.get("checkpoint_every_steps", 25)),
        "checkpoint_keep_last": train.get("checkpoint_keep_last", None),
        "ephemeral_checkpoint_every_steps": ephemeral,
        "pre_train_checkpoint": bool(train.get("pre_train_checkpoint", True)),
        "save_async": bool(train.get("save_async", True)),
        "sft_path": str(sft_path),
        "load_path": str(load_path),
        "max_examples": data.get("max_examples"),
        "save_folder": str(save_folder),
        "metrics_dir": scratch["metrics_dir"],
        "progress_dir": scratch["progress_dir"],
        "work_dir": scratch["work_dir"],
        "platform_run_id": platform_run_id(),
    }


def try_launch(plan: Dict[str, Any], cfg: Dict[str, Any], *, resume: bool) -> None:
    pin_cuda_visible_devices(cfg)
    soft_remote = bool(platform_run_id() and is_remote_uri(plan["save_folder"]))
    trainer_resume = True if soft_remote else resume
    try:
        from olmo_core.train import (  # type: ignore
            prepare_training_environment,
            teardown_training_environment,
        )
        from olmo_core.utils import seed_all  # type: ignore
    except ImportError as e:
        raise SystemExit(f"olmo_core not installed: {e}") from e

    from alt_cl.olmo_ext.sft_trainer_build import build_sft_trainer

    ensure_local_dir(plan["metrics_dir"])
    ensure_local_dir(plan["progress_dir"])
    ensure_local_dir(plan["work_dir"])
    cache = ensure_local_dir(Path(plan["work_dir"]).parent / "sft_cache")
    local_sft = _stage_sft_jsonl(plan["sft_path"], cache_dir=cache)

    prepare_training_environment(seed=int(plan["init_seed"]))
    seed_all(int(plan["init_seed"]))
    try:
        (Path(plan["progress_dir"]) / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": plan["run_id"],
                    "load_path": plan["load_path"],
                    "sft_path": plan["sft_path"],
                    "total_steps": plan["total_steps"],
                    "max_tokens": plan["max_tokens"],
                    "save_folder": plan["save_folder"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        trainer = build_sft_trainer(
            save_folder=plan["save_folder"],
            metrics_dir=plan["metrics_dir"],
            work_dir=plan["work_dir"],
            sft_jsonl_gz=local_sft,
            load_path=plan["load_path"],
            max_tokens=int(plan["max_tokens"]),
            global_batch_size=int(plan["global_batch_size"]),
            sequence_length=int(plan["sequence_length"]),
            lr=float(plan["lr"]),
            warmup_steps=int(plan["warmup_steps"]),
            lr_alpha_f=float(plan["lr_alpha_f"]),
            rank_microbatch_size=int(plan["rank_microbatch_size"]),
            seed=int(plan["seed"]),
            init_seed=int(plan["init_seed"]),
            model_arch=str(plan["model_arch"]),
            compile_model=bool(plan["compile_model"]),
            resume=trainer_resume,
            checkpoint_every_steps=int(plan["checkpoint_every_steps"]),
            checkpoint_keep_last=plan["checkpoint_keep_last"],
            ephemeral_checkpoint_every_steps=plan["ephemeral_checkpoint_every_steps"],
            pre_train_checkpoint=bool(plan["pre_train_checkpoint"]),
            save_async=bool(plan["save_async"]),
            max_examples=plan.get("max_examples"),
            tokenizer_id=str(plan["tokenizer"]),
        )
        print(
            json.dumps(
                {
                    "status": "fitting",
                    "run_id": plan["run_id"],
                    "load_path": plan["load_path"],
                    "sft_path": plan["sft_path"],
                    "total_steps": plan["total_steps"],
                    "max_tokens": plan["max_tokens"],
                    "save_folder": plan["save_folder"],
                    "resume": trainer_resume,
                },
                indent=2,
            ),
            flush=True,
        )
        trainer.fit()
    finally:
        teardown_training_environment()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "math_sft_60m.yaml",
    )
    ap.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Pretrain arm checkpoint dir (local or s3://) to finetune from",
    )
    ap.add_argument(
        "--sft-path",
        type=str,
        default=None,
        help="Local path or s3:// URI to train.jsonl.gz",
    )
    ap.add_argument(
        "--save-folder",
        type=str,
        default=None,
        help="Checkpoint dir (local or s3://). Defaults to $EDULLM_CHECKPOINT_DIR",
    )
    ap.add_argument("--olmo-root", type=Path, default=None)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    try:
        validate_sft_config(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    sft_raw = (
        args.sft_path
        or str((cfg.get("data") or {}).get("sft_path") or DEFAULT_SFT_PATH)
    )
    # Dry-run: only require local file when not remote.
    if not is_remote_uri(sft_raw) and not Path(sft_raw).is_file() and not args.launch:
        # Still allow dry-run plan if path is missing (document expected path).
        pass
    if args.launch and not is_remote_uri(sft_raw) and not Path(sft_raw).is_file():
        raise SystemExit(
            f"SFT mix not found: {sft_raw}\n"
            "Build with: python scripts/build_math_sft_60m.py"
        )

    if args.launch and args.olmo_root is None and not _olmo_core_importable():
        raise SystemExit(
            "--launch requires --olmo-root (or install olmo_core in the environment)"
        )
    if args.olmo_root is not None:
        rev = str((cfg.get("olmo_core") or {}).get("revision") or "")
        try:
            verify_olmo_revision(args.olmo_root, rev)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    save_folder = resolve_save_folder(cfg, ROOT, cli_save_folder=args.save_folder)
    scratch = resolve_run_scratch(cfg, ROOT, save_folder=save_folder)
    plan = build_plan(
        cfg,
        save_folder=save_folder,
        scratch=scratch,
        load_path=args.load_path,
        sft_path=sft_raw,
    )

    if args.launch:
        try_launch(plan, cfg, resume=args.resume)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
