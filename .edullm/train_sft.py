#!/usr/bin/env python3
"""Platform entry for alt-cl math SFT (identical recipe for both pretrain arms).

    bash -lc 'python -m torch.distributed.run --nproc-per-node=N --standalone \\
      .edullm/train_sft.py "$EDULLM_RUN_ID" \\
      --config configs/math_sft_60m.yaml \\
      --load-path "$PRETRAIN_CKPT" \\
      --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --sft-path s3://…/math-sft-60m/train.jsonl.gz'
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
        default=ROOT / "configs" / "math_sft_60m.yaml",
    )
    ap.add_argument(
        "--load-path",
        required=True,
        help="Pretrain checkpoint dir (usually s3://…/checkpoints/stepN)",
    )
    ap.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
    )
    ap.add_argument(
        "--sft-path",
        default=None,
        help="Local or s3:// path to train.jsonl.gz",
    )
    ap.add_argument("--olmo-root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"unrecognized arguments: {unknown}", file=sys.stderr)
        return 2

    if not args.dry_run and not str(args.save_folder or "").strip():
        print(
            "refusing to launch without --save-folder / $EDULLM_CHECKPOINT_DIR",
            file=sys.stderr,
        )
        return 64

    cmd = [
        sys.executable,
        "-m",
        "scripts.train_sft",
        "--config",
        str(args.config),
        "--load-path",
        str(args.load_path),
        "--save-folder",
        str(args.save_folder),
    ]
    if args.sft_path:
        cmd.extend(["--sft-path", str(args.sft_path)])
    if args.olmo_root is not None:
        cmd.extend(["--olmo-root", str(args.olmo_root)])
    if args.resume:
        cmd.append("--resume")
    if not args.dry_run:
        cmd.append("--launch")

    os.environ.setdefault("EDULLM_RUN_ID", str(args.run_name))
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
