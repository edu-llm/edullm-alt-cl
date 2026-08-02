"""Config load + experiment contract validation for alt-cl arms."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from alt_cl.data import DEFAULT_MATH_DATASET_ID, DEFAULT_REGMIX_DATASET_ID
from alt_cl.phases import (
    ANNEAL_MATH_STEPS,
    ANNEAL_WINDOW_STEPS,
    FRONT_MATH_STEPS,
    GLOBAL_BATCH_TOKENS,
    WARMUP_STEPS,
    phase_boundaries,
    total_steps_for_arm,
)
from alt_cl.platform_env import (  # noqa: F401 — re-export
    is_remote_uri,
    platform_run_id,
    resolve_output_dir,
    resolve_run_scratch,
    resolve_save_folder,
)

ALLOWED_ARMS = frozenset({"math-front-anneal", "math-anneal"})

# Platform outputs bucket (see edu-llm/platform contracts/results.py).
PLATFORM_OUTPUTS_BUCKET = "sbsandbox-intern-edullm-outputs"


def load_config(path: Path | str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML is required (pip install PyYAML)") from e
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"config root must be a mapping: {p}")
    return data


def s3_checkpoint_uri(cfg: Mapping[str, Any], *parts: str) -> str:
    """Informational URI for dry-run plans.

    On the platform, the real checkpoint location is ``$EDULLM_CHECKPOINT_DIR``.
    YAML ``s3.prefix`` is only a legacy label when that env is unset.
    """
    env = (os.environ.get("EDULLM_CHECKPOINT_DIR") or "").strip()
    if env:
        extra = "/".join(p.strip("/") for p in parts if str(p).strip())
        return f"{env.rstrip('/')}/{extra}" if extra else env
    s3 = cfg.get("s3") or {}
    bucket = str(
        s3.get("checkpoint_bucket") or PLATFORM_OUTPUTS_BUCKET
    ).strip()
    prefix = str(s3.get("prefix") or "").strip().strip("/")
    extra = "/".join(p.strip("/") for p in parts if str(p).strip())
    base = f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    return f"{base}/{extra}" if extra else base


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Fail closed on missing / inconsistent alt-cl identity fields."""
    run_id = str(cfg.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    arm = str(cfg.get("arm") or "").strip().lower().replace("_", "-")
    if arm not in ALLOWED_ARMS:
        raise ValueError(f"arm must be one of {sorted(ALLOWED_ARMS)}, got {arm!r}")
    if cfg.get("seed") is None:
        raise ValueError("seed is required")

    data = cfg.get("data") or {}
    if not data.get("tokenizer"):
        raise ValueError("data.tokenizer is required")
    if int(data.get("sequence_length") or 0) <= 0:
        raise ValueError("data.sequence_length must be > 0")
    regmix_id = str(data.get("regmix_dataset_id") or DEFAULT_REGMIX_DATASET_ID)
    math_id = str(data.get("math_dataset_id") or DEFAULT_MATH_DATASET_ID)
    if "/" not in regmix_id or "/" not in math_id:
        raise ValueError("dataset ids must be family/name (e.g. pretrain/regmix-10b)")

    model = cfg.get("model") or {}
    if not model.get("arch"):
        raise ValueError("model.arch is required (e.g. olmo2_370M)")
    if model.get("load_path"):
        raise ValueError("model.load_path must be null (scratch only)")

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
            f"global_batch_size ({gbs}); got remainder {max_tokens % gbs}"
        )
    if gbs != GLOBAL_BATCH_TOKENS and arm in ALLOWED_ARMS:
        # Allow smoke overrides; warn via ValueError only if not divisible by seq
        pass
    seq = int(data["sequence_length"])
    if gbs % seq != 0:
        raise ValueError(
            f"global_batch_size {gbs} must be divisible by sequence_length {seq}"
        )

    phases = cfg.get("phases") or {}
    warmup = int(phases.get("warmup_steps", train.get("warmup_steps", WARMUP_STEPS)))
    front = int(phases.get("front_math_steps", FRONT_MATH_STEPS))
    anneal_w = int(phases.get("anneal_window_steps", ANNEAL_WINDOW_STEPS))
    anneal_m = int(phases.get("anneal_math_steps", ANNEAL_MATH_STEPS))
    total_steps = max_tokens // gbs
    arm_key = arm
    if arm_key == "math-front-anneal":
        regmix_steps = total_steps - front
    else:
        regmix_steps = total_steps
    if regmix_steps <= 0:
        raise ValueError("schedule leaves no regmix steps")
    built = phase_boundaries(
        arm,  # type: ignore[arg-type]
        warmup_steps=warmup,
        front_math_steps=front,
        regmix_steps=regmix_steps,
        anneal_window_steps=anneal_w,
        anneal_math_steps=anneal_m,
    )
    if sum(p.n_steps for p in built) != total_steps:
        raise ValueError(
            f"phase steps {sum(p.n_steps for p in built)} != max_tokens/GBS {total_steps}"
        )


def verify_olmo_revision(olmo_root: Path | str, expected: str) -> None:
    root = Path(olmo_root)
    if not root.is_dir():
        raise RuntimeError(f"olmo-root not a directory: {root}")
    try:
        got = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to read git HEAD under {root}: {exc}") from exc
    if got != expected.strip():
        raise RuntimeError(
            f"OLMo-core revision mismatch under {root}: got {got}, expected {expected}"
        )


def run_fingerprint(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Scientific identity that must match on --resume."""
    data = cfg.get("data") or {}
    train = cfg.get("train") or {}
    phases = cfg.get("phases") or {}
    model = cfg.get("model") or {}
    olmo = cfg.get("olmo_core") or {}
    arm = str(cfg.get("arm") or "").strip().lower().replace("_", "-")
    return {
        "run_id": str(cfg["run_id"]),
        "arm": arm,
        "seed": int(cfg["seed"]),
        "init_seed": int(model.get("init_seed", cfg["seed"])),
        "model_arch": str(model.get("arch")),
        "olmo_revision": str(olmo.get("revision") or ""),
        "tokenizer": str(data.get("tokenizer")),
        "sequence_length": int(data.get("sequence_length")),
        "max_tokens": int(train.get("max_tokens")),
        "global_batch_size": int(train.get("global_batch_size")),
        "lr": float(train.get("lr")),
        "warmup_steps": int(
            phases.get("warmup_steps", train.get("warmup_steps", WARMUP_STEPS))
        ),
        "front_math_steps": int(phases.get("front_math_steps", FRONT_MATH_STEPS)),
        "anneal_window_steps": int(
            phases.get("anneal_window_steps", ANNEAL_WINDOW_STEPS)
        ),
        "anneal_math_steps": int(phases.get("anneal_math_steps", ANNEAL_MATH_STEPS)),
        "regmix_dataset_id": str(
            data.get("regmix_dataset_id") or DEFAULT_REGMIX_DATASET_ID
        ),
        "math_dataset_id": str(data.get("math_dataset_id") or DEFAULT_MATH_DATASET_ID),
    }


def fingerprint_sha256(fp: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(fp), sort_keys=True).encode("utf-8")
    ).hexdigest()


def compare_fingerprints(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> Optional[str]:
    """Return None if compatible; else an error message.

    Extending ``max_tokens`` upward is the only allowed identity change on resume.
    """
    if dict(prior) == dict(current):
        return None
    diffs = sorted(k for k in set(prior) | set(current) if prior.get(k) != current.get(k))
    if diffs == ["max_tokens"] and int(current["max_tokens"]) > int(prior["max_tokens"]):
        return None
    return (
        "Refusing to resume: run identity changed "
        f"(differing fields: {diffs}). Extending max_tokens upward is the only "
        "allowed identity change on --resume."
    )


def default_total_steps_for_config(cfg: Mapping[str, Any]) -> int:
    arm = str(cfg.get("arm") or "").strip().lower().replace("_", "-")
    train = cfg.get("train") or {}
    gbs = int(train.get("global_batch_size") or GLOBAL_BATCH_TOKENS)
    max_tokens = int(train.get("max_tokens") or 0)
    if max_tokens > 0:
        return max_tokens // gbs
    return total_steps_for_arm(arm)  # type: ignore[arg-type]
