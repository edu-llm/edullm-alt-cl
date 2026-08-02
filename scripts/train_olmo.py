#!/usr/bin/env python3
"""OLMo-core entry for alt-cl math front-load / end-anneal arms.

Requires edu-llm/OLMo-core installed (revision pinned in the YAML). Builds a
real Trainer + CurriculumDataLoader from the arm config.

  # Dry-run plan:
  python -m scripts.train_olmo --config configs/math_front_anneal_10b.yaml

  # Launch:
  CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \\
    --nproc_per_node=1 -m scripts.train_olmo \\
    --config configs/math_front_anneal_10b.yaml \\
    --olmo-root /path/to/OLMo-core --launch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alt_cl.contract import (  # noqa: E402
    compare_fingerprints,
    load_config,
    resolve_run_scratch,
    resolve_save_folder,
    run_fingerprint,
    s3_checkpoint_uri,
    validate_config,
    verify_olmo_revision,
)
from alt_cl.data import (  # noqa: E402
    DEFAULT_MATH_DATASET_ID,
    DEFAULT_REGMIX_DATASET_ID,
    read_paths_file,
    resolve_and_stage_train_tokens,
    write_stage_meta,
)
from alt_cl.platform_env import (  # noqa: E402
    ensure_local_dir,
    is_remote_uri,
    join_uri,
    local_dir_nonempty,
    platform_run_id,
    read_json_uri,
    write_json_uri,
)

log = logging.getLogger("scripts.train_olmo")

_IDLE_MEMORY_MIB = 256


def pin_cuda_visible_devices(cfg: Dict[str, Any]) -> Optional[str]:
    """If ``train.cuda_visible_devices`` is set, pin and require an idle GPU."""
    raw = (cfg.get("train") or {}).get("cuda_visible_devices")
    if raw is None or str(raw).strip() == "":
        return None
    pinned = str(raw).strip()
    if "," in pinned or not pinned.isdigit():
        raise SystemExit(
            f"train.cuda_visible_devices must be a single integer GPU index, got {pinned!r}"
        )
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and existing.strip() != pinned:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES is already {existing!r} but "
            f"train.cuda_visible_devices is {pinned!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = pinned
    try:
        probe = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                pinned,
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"Unable to query physical GPU {pinned} via nvidia-smi ({exc})"
        ) from exc
    parts = [p.strip() for p in probe.split(",")]
    if len(parts) < 3 or parts[0] != pinned:
        raise SystemExit(f"Unexpected nvidia-smi response for GPU {pinned}: {probe!r}")
    used_mib = int(float(parts[2]))
    if used_mib > _IDLE_MEMORY_MIB:
        raise SystemExit(
            f"Physical GPU {pinned} is not idle ({used_mib} MiB used > "
            f"{_IDLE_MEMORY_MIB} MiB). Refusing to launch."
        )
    print(
        json.dumps(
            {
                "cuda_visible_devices": pinned,
                "uuid": parts[1],
                "memory_used_mib": used_mib,
                "status": "pinned_idle",
            }
        ),
        flush=True,
    )
    return pinned


def _stage_corpus(
    *,
    label: str,
    dataset_id: str,
    version: Optional[str],
    paths_file: Optional[str],
    cache_dir: Path,
    allow_missing: bool,
) -> Tuple[Optional[List[str]], Optional[str], Any]:
    import numpy as np

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


def build_plan(
    cfg: Dict[str, Any],
    *,
    out: Path,
    save_folder: str,
    scratch: Dict[str, str],
) -> Dict[str, Any]:
    train = cfg.get("train") or {}
    data = cfg.get("data") or {}
    phases = cfg.get("phases") or {}
    model = cfg.get("model") or {}
    olmo = cfg.get("olmo_core") or {}
    arm = str(cfg["arm"]).strip().lower().replace("_", "-")
    max_tokens = int(train["max_tokens"])
    gbs = int(train["global_batch_size"])
    total_steps = max_tokens // gbs
    ephemeral = train.get("ephemeral_checkpoint_every_steps", None)
    return {
        "run_id": cfg["run_id"],
        "arm": arm,
        "seed": int(cfg["seed"]),
        "init_seed": int(model.get("init_seed", cfg["seed"])),
        "model_name": model.get("name"),
        "model_arch": str(model["arch"]),
        "olmo_revision": str(olmo.get("revision") or ""),
        "tokenizer": str(data["tokenizer"]),
        "sequence_length": int(data["sequence_length"]),
        "max_tokens": max_tokens,
        "global_batch_size": gbs,
        "total_steps": total_steps,
        "lr": float(train["lr"]),
        "warmup_steps": int(
            phases.get("warmup_steps", train.get("warmup_steps", 24))
        ),
        "front_math_steps": int(phases.get("front_math_steps", 24)),
        "anneal_window_steps": int(phases.get("anneal_window_steps", 119)),
        "anneal_math_steps": int(phases.get("anneal_math_steps", 24)),
        "lr_alpha_f": float(train.get("lr_alpha_f", 1.0)),
        "rank_microbatch_size": int(train.get("rank_microbatch_size", 65536)),
        "num_workers": int(train.get("num_workers", 4)),
        "compile_model": bool(train.get("compile_model", True)),
        "checkpoint_every_steps": int(train.get("checkpoint_every_steps", 125)),
        "checkpoint_keep_last": train.get("checkpoint_keep_last", None),
        "ephemeral_checkpoint_every_steps": ephemeral,
        "pre_train_checkpoint": bool(train.get("pre_train_checkpoint", True)),
        "save_async": bool(train.get("save_async", True)),
        "regmix_dataset_id": str(
            data.get("regmix_dataset_id") or DEFAULT_REGMIX_DATASET_ID
        ),
        "math_dataset_id": str(data.get("math_dataset_id") or DEFAULT_MATH_DATASET_ID),
        "allow_missing_math": bool(data.get("allow_missing_math", False)),
        "save_folder": str(save_folder),
        "metrics_dir": scratch["metrics_dir"],
        "progress_dir": scratch["progress_dir"],
        "dataset_cache": scratch["dataset_cache"],
        "work_dir": scratch["work_dir"],
        "s3_checkpoints": s3_checkpoint_uri(cfg),
        "fingerprint": run_fingerprint(cfg),
        "platform_run_id": platform_run_id(),
    }


def _fingerprint_uri(plan: Dict[str, Any]) -> str:
    return join_uri(str(plan["save_folder"]), "run_fingerprint.json")


def _assert_fresh_save_folder(save_folder: str) -> None:
    if is_remote_uri(save_folder):
        # Platform Batch retries reuse the same EDULLM_CHECKPOINT_DIR on purpose.
        return
    if local_dir_nonempty(save_folder):
        contents = sorted(p.name for p in Path(save_folder).iterdir())[:5]
        raise SystemExit(
            f"Scratch run refuses non-empty save folder: {save_folder} "
            f"(contains {contents}). Pass --resume, choose a new output_dir, "
            "or clear the folder."
        )


def _prepare_run_dir(
    plan: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    resume: bool,
    soft_remote: bool = False,
) -> None:
    save_folder = str(plan["save_folder"])
    fp_uri = _fingerprint_uri(plan)
    current = run_fingerprint(cfg)
    if soft_remote:
        # Platform retries reuse the prefix; first attempt has no fingerprint yet.
        prior = read_json_uri(fp_uri)
        if prior is not None:
            err = compare_fingerprints(prior, current)
            if err:
                raise SystemExit(err)
        return
    if resume:
        prior = read_json_uri(fp_uri)
        if prior is None:
            raise SystemExit(
                f"--resume set but no run_fingerprint.json under {save_folder}"
            )
        err = compare_fingerprints(prior, current)
        if err:
            raise SystemExit(err)
        return
    _assert_fresh_save_folder(save_folder)
    if not is_remote_uri(save_folder):
        Path(save_folder).mkdir(parents=True, exist_ok=True)


def _commit_run_fingerprint(plan: Dict[str, Any], cfg: Dict[str, Any], *, resume: bool) -> None:
    fp_uri = _fingerprint_uri(plan)
    current = run_fingerprint(cfg)
    if resume:
        prior = read_json_uri(fp_uri)
        if prior is not None:
            if prior == current:
                return
            err = compare_fingerprints(prior, current)
            if err:
                raise SystemExit(err)
    write_json_uri(fp_uri, current)


def _olmo_core_importable() -> bool:
    try:
        import olmo_core  # noqa: F401

        return True
    except ImportError:
        return False


def try_launch(plan: Dict[str, Any], cfg: Dict[str, Any], *, resume: bool) -> None:
    pin_cuda_visible_devices(cfg)
    # Platform Batch retries share EDULLM_CHECKPOINT_DIR — soft-resume + if_available.
    soft_remote = bool(platform_run_id() and is_remote_uri(plan["save_folder"]))
    trainer_resume = True if soft_remote else resume
    _prepare_run_dir(plan, cfg, resume=resume, soft_remote=soft_remote)

    try:
        from olmo_core.train import (  # type: ignore
            prepare_training_environment,
            teardown_training_environment,
        )
        from olmo_core.utils import seed_all  # type: ignore
        import torch
    except ImportError as e:
        raise SystemExit(f"olmo_core not installed: {e}") from e

    from alt_cl.olmo_ext.trainer_build import build_trainer

    data = cfg.get("data") or {}
    cache_dir = ensure_local_dir(plan["dataset_cache"])
    ensure_local_dir(plan["metrics_dir"])
    ensure_local_dir(plan["progress_dir"])
    ensure_local_dir(plan["work_dir"])

    prepare_training_environment(seed=int(plan["init_seed"]))
    seed_all(int(plan["init_seed"]))
    try:
        # Stage on rank 0 after env init so dist is ready for barriers if needed.
        from olmo_core.distributed.utils import get_rank, is_distributed
        import torch.distributed as dist

        rank = get_rank()
        if rank == 0:
            regmix_paths, regmix_ver, regmix_dtype = _stage_corpus(
                label="regmix",
                dataset_id=plan["regmix_dataset_id"],
                version=data.get("regmix_version"),
                paths_file=data.get("regmix_paths_file"),
                cache_dir=cache_dir,
                allow_missing=False,
            )
            assert regmix_paths is not None
            math_paths, math_ver, math_dtype = _stage_corpus(
                label="math",
                dataset_id=plan["math_dataset_id"],
                version=data.get("math_version"),
                paths_file=data.get("math_paths_file"),
                cache_dir=cache_dir,
                allow_missing=bool(plan["allow_missing_math"]),
            )
            stage = {
                "regmix": {
                    "dataset_id": plan["regmix_dataset_id"],
                    "version": regmix_ver,
                    "paths": regmix_paths,
                    "dtype": str(regmix_dtype),
                },
                "math": {
                    "dataset_id": plan["math_dataset_id"],
                    "version": math_ver,
                    "paths": math_paths,
                    "dtype": str(math_dtype) if math_dtype is not None else None,
                },
                "arm": plan["arm"],
            }
            (cache_dir / "_stage_alt_cl.json").write_text(
                json.dumps(stage, indent=2) + "\n", encoding="utf-8"
            )
            (Path(plan["progress_dir"]) / "run_meta.json").write_text(
                json.dumps(
                    {
                        "run_id": plan["run_id"],
                        "arm": plan["arm"],
                        "total_steps": plan["total_steps"],
                        "regmix": stage["regmix"]["dataset_id"],
                        "math": stage["math"]["dataset_id"],
                        "architecture": plan["model_name"],
                        "save_folder": plan["save_folder"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if is_distributed():
            dist.barrier()

        stage = json.loads((cache_dir / "_stage_alt_cl.json").read_text(encoding="utf-8"))
        import numpy as np

        trainer = build_trainer(
            arm=plan["arm"],
            save_folder=plan["save_folder"],
            metrics_dir=plan["metrics_dir"],
            work_dir=plan["work_dir"],
            regmix_paths=list(stage["regmix"]["paths"]),
            math_paths=list(stage["math"]["paths"]) if stage["math"]["paths"] else None,
            regmix_dtype=np.dtype(stage["regmix"]["dtype"] or "uint32"),
            math_dtype=np.dtype(stage["math"]["dtype"] or "uint32")
            if stage["math"]["dtype"]
            else np.uint32,
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
            num_workers=int(plan["num_workers"]),
            resume=trainer_resume,
            front_math_steps=int(plan["front_math_steps"]),
            anneal_window_steps=int(plan["anneal_window_steps"]),
            anneal_math_steps=int(plan["anneal_math_steps"]),
            checkpoint_every_steps=int(plan["checkpoint_every_steps"]),
            checkpoint_keep_last=plan["checkpoint_keep_last"],
            ephemeral_checkpoint_every_steps=plan["ephemeral_checkpoint_every_steps"],
            pre_train_checkpoint=bool(plan["pre_train_checkpoint"]),
            save_async=bool(plan["save_async"]),
        )
        _commit_run_fingerprint(plan, cfg, resume=trainer_resume)
        print(
            json.dumps(
                {
                    "status": "fitting",
                    "arm": plan["arm"],
                    "run_id": plan["run_id"],
                    "resume": trainer_resume,
                    "save_folder": plan["save_folder"],
                    "max_tokens": plan["max_tokens"],
                    "total_steps": plan["total_steps"],
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
        default=ROOT / "configs" / "math_front_anneal_10b.yaml",
    )
    ap.add_argument(
        "--olmo-root",
        type=Path,
        default=None,
        help="Pinned OLMo-core checkout; optional if olmo_core is already importable",
    )
    ap.add_argument(
        "--save-folder",
        type=str,
        default=None,
        help="Checkpoint dir (local or s3://). Defaults to $EDULLM_CHECKPOINT_DIR "
        "then <output_dir>/checkpoints",
    )
    ap.add_argument(
        "--launch",
        action="store_true",
        help="Build trainer and call fit()",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume if run_fingerprint.json matches",
    )
    ap.add_argument(
        "--data-cache-dir",
        type=Path,
        default=None,
        help="Override dataset staging cache (defaults under output_dir)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    try:
        validate_config(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.launch and args.olmo_root is None and not _olmo_core_importable():
        raise SystemExit(
            "--launch requires --olmo-root pointing at the pinned OLMo-core checkout "
            "(or install olmo_core in the environment)"
        )
    if args.olmo_root is not None:
        rev = str((cfg.get("olmo_core") or {}).get("revision") or "")
        try:
            verify_olmo_revision(args.olmo_root, rev)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    save_folder = resolve_save_folder(cfg, ROOT, cli_save_folder=args.save_folder)
    scratch = resolve_run_scratch(cfg, ROOT, save_folder=save_folder)
    plan = build_plan(cfg, out=Path(scratch["output_dir"]), save_folder=save_folder, scratch=scratch)
    if args.data_cache_dir is not None:
        plan["dataset_cache"] = str(args.data_cache_dir)

    if args.launch:
        try_launch(plan, cfg, resume=args.resume)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
