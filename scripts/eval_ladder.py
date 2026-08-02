#!/usr/bin/env python3
"""Run the alt-cl GSM8K / MATH eval ladder via olmo-eval-full.

Default is a dry-run plan JSON (no GPU, no olmo-eval install required).

  python -m scripts.eval_ladder --config configs/eval_math_ladder.yaml \\
      --arm math-front-anneal \\
      --checkpoint-dir /path/to/sft/checkpoints \\
      --base-checkpoint /path/to/pretrain/checkpoints/step2408

  # Execute (olmo-eval on PATH, GPU host / olmo-eval-full image):
  python -m scripts.eval_ladder … --launch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alt_cl.contract import load_config  # noqa: E402
from alt_cl.eval_ladder import (  # noqa: E402
    build_eval_plan,
    resolve_harness_config_path,
    validate_eval_config,
)
from alt_cl.platform_env import (  # noqa: E402
    ensure_local_dir,
    is_remote_uri,
    platform_run_id,
    resolve_output_dir,
)

log = logging.getLogger("scripts.eval_ladder")


def _parse_steps(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval_math_ladder.yaml",
    )
    ap.add_argument(
        "--arm",
        required=True,
        help="Arm label for the plan (math-front-anneal | math-anneal)",
    )
    ap.add_argument(
        "--checkpoint-dir",
        required=True,
        help="SFT Checkpointer save_folder (local or s3://…/checkpoints)",
    )
    ap.add_argument(
        "--base-checkpoint",
        default=None,
        help="Pretrain final checkpoint dir (stepN), scored as stage=base",
    )
    ap.add_argument(
        "--steps",
        default=None,
        help="Comma-separated SFT steps to eval (default: ladder from config)",
    )
    ap.add_argument(
        "--include-optional",
        action="store_true",
        help="Also run tasks_optional (e.g. gsm_symbolic)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Results root (default: YAML output_dir/metrics or EDULLM_OUTPUT_PREFIX/eval)",
    )
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument(
        "--olmo-eval-bin",
        default=None,
        help="Override harness.cli (default: olmo-eval)",
    )
    ap.add_argument(
        "--launch",
        action="store_true",
        help="Execute olmo-eval commands (default: print plan JSON only)",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="With --launch, keep going if one job fails",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    validate_eval_config(cfg)

    harness = dict(cfg.get("harness") or {})
    if args.olmo_eval_bin:
        harness["cli"] = str(args.olmo_eval_bin)
        cfg = dict(cfg)
        cfg["harness"] = harness

    harness_path = resolve_harness_config_path(cfg, ROOT)
    if not harness_path.is_file():
        raise SystemExit(f"harness config not found: {harness_path}")

    if args.output_dir:
        output_dir = str(args.output_dir).strip()
    else:
        env_prefix = (os.environ.get("EDULLM_OUTPUT_PREFIX") or "").strip()
        if env_prefix:
            output_dir = f"{env_prefix.rstrip('/')}/eval"
        else:
            out = resolve_output_dir(cfg, ROOT)
            output_dir = str(out / "eval")

    dry_run = not args.launch
    plan = build_eval_plan(
        cfg,
        root=ROOT,
        arm=args.arm,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=output_dir,
        base_checkpoint=args.base_checkpoint,
        steps=_parse_steps(args.steps),
        include_optional=bool(args.include_optional),
        dry_run=dry_run,
        num_gpus=int(args.num_gpus),
    )
    plan["platform_run_id"] = platform_run_id()
    print(json.dumps(plan, indent=2))

    if dry_run:
        return 0

    cli = str(harness.get("cli") or "olmo-eval")
    if shutil.which(cli) is None and not Path(cli).is_file():
        print(
            f"olmo-eval binary not found: {cli!r}\n"
            "Install / use the edu-llm/olmo-eval-full image at harness.revision, "
            "or pass --olmo-eval-bin.",
            file=sys.stderr,
        )
        return 127

    if not is_remote_uri(output_dir):
        ensure_local_dir(output_dir)

    failures = 0
    for i, cmd in enumerate(plan["commands"]):
        job = plan["jobs"][i]
        log.info("eval job %s: %s", job["label"], " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures += 1
            log.error("job %s failed with exit %s", job["label"], proc.returncode)
            if not args.continue_on_error:
                return proc.returncode or 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
