#!/usr/bin/env python3
"""Platform entry for the alt-cl GSM8K / MATH eval ladder.

Prefer submitting against the registered ``olmo-eval-full`` image when possible.
This entry exists so an ``edullm-alt-cl`` workload can orchestrate the ladder
once that repo is registered, shelling out to ``olmo-eval`` installed in the
image (or a sibling mount).

    bash -lc 'python .edullm/eval_ladder.py "$EDULLM_RUN_ID" \\
      --config configs/eval_math_ladder.yaml \\
      --arm math-front-anneal \\
      --checkpoint-dir "$SFT_CKPT_DIR" \\
      --base-checkpoint "$PRETRAIN_FINAL" \\
      --output-dir "$EDULLM_OUTPUT_PREFIX/eval"'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_name",
        nargs="?",
        default=os.environ.get("EDULLM_RUN_ID", "local"),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval_math_ladder.yaml",
    )
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--steps", default=None)
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to $EDULLM_OUTPUT_PREFIX/eval when set",
    )
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--olmo-eval-bin", default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan JSON only (no olmo-eval fit)",
    )
    ap.add_argument("--continue-on-error", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"unrecognized arguments: {unknown}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval_ladder",
        "--config",
        str(args.config),
        "--arm",
        str(args.arm),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--num-gpus",
        str(int(args.num_gpus)),
    ]
    if args.base_checkpoint:
        cmd.extend(["--base-checkpoint", str(args.base_checkpoint)])
    if args.steps:
        cmd.extend(["--steps", str(args.steps)])
    if args.include_optional:
        cmd.append("--include-optional")
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    elif (os.environ.get("EDULLM_OUTPUT_PREFIX") or "").strip():
        prefix = os.environ["EDULLM_OUTPUT_PREFIX"].rstrip("/")
        cmd.extend(["--output-dir", f"{prefix}/eval"])
    if args.olmo_eval_bin:
        cmd.extend(["--olmo-eval-bin", str(args.olmo_eval_bin)])
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if not args.dry_run:
        cmd.append("--launch")

    os.environ.setdefault("EDULLM_RUN_ID", str(args.run_name))
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
