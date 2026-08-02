"""Eval ladder plan: GSM8K / MATH vs SFT step for alt-cl arms.

Builds ``olmo-eval`` command lines against raw OLMo-core checkpoints via the
``olmo_core`` provider. Does not import ``olmo_eval`` — that lives in the
pinned ``edu-llm/olmo-eval-full`` image / checkout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from alt_cl.checkpoint_ladder import permanent_checkpoint_steps
from alt_cl.platform_env import is_remote_uri, join_uri

DEFAULT_EVAL_HARNESS_REV = "37e81867d4d71c7c7343c63a3590f79bd43987c4"
DEFAULT_CHECKPOINT_DIRNAME = "step{step}"
PRIMARY_TASK = "gsm8k"


@dataclass(frozen=True)
class EvalJob:
    """One (checkpoint × task-set) evaluation."""

    arm: str
    stage: str  # "base" | "sft"
    step: Optional[int]
    checkpoint: str
    tasks: tuple[str, ...]
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_eval_config(cfg: Mapping[str, Any]) -> None:
    run_id = str(cfg.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("run_id is required")
    harness = cfg.get("harness") or {}
    if not str(harness.get("revision") or "").strip():
        raise ValueError("harness.revision is required (pin olmo-eval-full)")
    kind = str(harness.get("provider_kind") or "").strip()
    if kind != "olmo_core":
        raise ValueError(
            f"harness.provider_kind must be 'olmo_core' for raw Trainer ckpts, got {kind!r}"
        )
    if not str(harness.get("harness_config") or "").strip():
        raise ValueError("harness.harness_config is required")
    tasks = list(cfg.get("tasks") or [])
    if not tasks:
        raise ValueError("tasks must be a non-empty list (e.g. [gsm8k, minerva_math])")
    if PRIMARY_TASK not in tasks:
        raise ValueError(f"tasks must include primary readout {PRIMARY_TASK!r}")
    ladder = cfg.get("ladder") or {}
    total = int(ladder.get("sft_total_steps") or 0)
    every = int(ladder.get("sft_checkpoint_every") or 0)
    if total <= 0 or every <= 0:
        raise ValueError("ladder.sft_total_steps and sft_checkpoint_every must be > 0")


def sft_eval_steps(
    total_steps: int,
    interval: int,
    *,
    include_zero: bool = False,
) -> List[int]:
    """Permanent SFT save steps to score (matches CheckpointerCallback ladder)."""
    steps = permanent_checkpoint_steps(int(total_steps), interval=int(interval))
    if not include_zero:
        steps = [s for s in steps if s != 0]
    return steps


def checkpoint_dirname(step: int, template: str = DEFAULT_CHECKPOINT_DIRNAME) -> str:
    return str(template).format(step=int(step))


def join_checkpoint(save_folder: str, step: int, *, template: str = DEFAULT_CHECKPOINT_DIRNAME) -> str:
    """``save_folder/stepN`` for local or ``s3://`` roots."""
    name = checkpoint_dirname(step, template)
    base = str(save_folder).rstrip("/")
    if is_remote_uri(base):
        return join_uri(base, name)
    return str(Path(base) / name)


def resolve_harness_config_path(cfg: Mapping[str, Any], root: Path) -> Path:
    raw = str((cfg.get("harness") or {}).get("harness_config") or "").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return p


def build_eval_jobs(
    cfg: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_dir: str,
    base_checkpoint: Optional[str] = None,
    steps: Optional[Sequence[int]] = None,
    include_optional: bool = False,
) -> List[EvalJob]:
    """Expand base + SFT ladder into concrete eval jobs."""
    validate_eval_config(cfg)
    arm_name = str(arm).strip().lower().replace("_", "-")
    allowed = {
        str(a).strip().lower().replace("_", "-")
        for a in (cfg.get("arms") or [])
        if str(a).strip()
    }
    if allowed and arm_name not in allowed:
        raise ValueError(f"arm {arm_name!r} not in config arms {sorted(allowed)}")

    tasks = tuple(str(t) for t in (cfg.get("tasks") or []))
    if include_optional:
        extra = tuple(str(t) for t in (cfg.get("tasks_optional") or []))
        tasks = tuple(dict.fromkeys(tasks + extra))

    ladder = cfg.get("ladder") or {}
    template = str(
        ladder.get("checkpoint_dirname_template") or DEFAULT_CHECKPOINT_DIRNAME
    )
    if steps is None:
        step_list = sft_eval_steps(
            int(ladder["sft_total_steps"]),
            int(ladder["sft_checkpoint_every"]),
            include_zero=bool(ladder.get("include_sft_step_zero", False)),
        )
    else:
        step_list = [int(s) for s in steps]

    jobs: List[EvalJob] = []
    if base_checkpoint:
        base = str(base_checkpoint).strip()
        if not base:
            raise ValueError("base_checkpoint is empty")
        jobs.append(
            EvalJob(
                arm=arm_name,
                stage="base",
                step=None,
                checkpoint=base,
                tasks=tasks,
                label=f"{arm_name}/base",
            )
        )

    ckpt_root = str(checkpoint_dir).strip()
    if not ckpt_root:
        raise ValueError("checkpoint_dir is required")
    for step in step_list:
        jobs.append(
            EvalJob(
                arm=arm_name,
                stage="sft",
                step=int(step),
                checkpoint=join_checkpoint(ckpt_root, int(step), template=template),
                tasks=tasks,
                label=f"{arm_name}/sft/step{int(step)}",
            )
        )
    if not jobs:
        raise ValueError("no eval jobs: pass --base-checkpoint and/or SFT steps")
    return jobs


def build_olmo_eval_argv(
    job: EvalJob,
    *,
    harness_config: Path | str,
    output_dir: str,
    cli: str = "olmo-eval",
    dry_run: bool = False,
    num_gpus: int = 1,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Argv for one ``olmo-eval run`` invocation."""
    out = str(output_dir).rstrip("/")
    if is_remote_uri(out):
        job_out = join_uri(out, job.label)
    else:
        job_out = str(Path(out) / job.label)

    argv: List[str] = [
        str(cli),
        "run",
        "-m",
        job.checkpoint,
        "--harness-config",
        str(harness_config),
        "-O",
        job_out,
        "--num-gpus",
        str(int(num_gpus)),
        "--experiment-name",
        job.label,
    ]
    for task in job.tasks:
        argv.extend(["-t", task])
    if dry_run:
        argv.append("--dry-run")
    if extra_args:
        argv.extend(str(a) for a in extra_args)
    return argv


def build_eval_plan(
    cfg: Mapping[str, Any],
    *,
    root: Path,
    arm: str,
    checkpoint_dir: str,
    output_dir: str,
    base_checkpoint: Optional[str] = None,
    steps: Optional[Sequence[int]] = None,
    include_optional: bool = False,
    dry_run: bool = True,
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """JSON-serializable plan (default dry-run surface for scripts.eval_ladder)."""
    validate_eval_config(cfg)
    harness = cfg.get("harness") or {}
    harness_path = resolve_harness_config_path(cfg, root)
    jobs = build_eval_jobs(
        cfg,
        arm=arm,
        checkpoint_dir=checkpoint_dir,
        base_checkpoint=base_checkpoint,
        steps=steps,
        include_optional=include_optional,
    )
    cli = str(harness.get("cli") or "olmo-eval")
    commands = [
        build_olmo_eval_argv(
            job,
            harness_config=harness_path,
            output_dir=output_dir,
            cli=cli,
            dry_run=dry_run,
            num_gpus=num_gpus,
        )
        for job in jobs
    ]
    ladder = cfg.get("ladder") or {}
    return {
        "run_id": str(cfg.get("run_id")),
        "arm": str(arm).strip().lower().replace("_", "-"),
        "seed": cfg.get("seed"),
        "harness_repository": harness.get("repository"),
        "harness_revision": harness.get("revision"),
        "provider_kind": harness.get("provider_kind"),
        "harness_config": str(harness_path),
        "tokenizer": harness.get("tokenizer"),
        "tasks": [str(t) for t in (cfg.get("tasks") or [])],
        "tasks_optional": [str(t) for t in (cfg.get("tasks_optional") or [])],
        "include_optional": bool(include_optional),
        "sft_total_steps": int(ladder.get("sft_total_steps") or 0),
        "sft_checkpoint_every": int(ladder.get("sft_checkpoint_every") or 0),
        "checkpoint_dir": checkpoint_dir,
        "base_checkpoint": base_checkpoint,
        "output_dir": output_dir,
        "primary_metric": "gsm8k accuracy vs SFT step",
        "jobs": [j.to_dict() for j in jobs],
        "commands": commands,
        "dry_run": bool(dry_run),
        "note": (
            "Requires olmo-eval-full @ harness.revision with olmo_core provider; "
            "pass --launch to execute commands (needs GPU + olmo-eval on PATH)."
        ),
    }
