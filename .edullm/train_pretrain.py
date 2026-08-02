#!/usr/bin/env python3
"""Platform entry for alt-cl pretrain arms.

    bash -lc 'python -m torch.distributed.run --nproc-per-node=N --standalone \\
      .edullm/train_pretrain.py "$EDULLM_RUN_ID" \\
      --config configs/math_front_anneal_10b.yaml \\
      --save-folder "$EDULLM_CHECKPOINT_DIR"'

Resolves both regmix + math from the YAML via edullm_data (form dataset_release: none).
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
        help="Run id (platform sets EDULLM_RUN_ID)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "math_front_anneal_10b.yaml",
    )
    ap.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        help="Must be $EDULLM_CHECKPOINT_DIR on platform (visible to checkpoint guard)",
    )
    ap.add_argument("--olmo-root", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan JSON only (no --launch)",
    )
    ap.add_argument("--data-cache-dir", type=Path, default=None)
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        print(f"unrecognized arguments: {unknown}", file=sys.stderr)
        return 2

    if not args.dry_run and not str(args.save_folder or "").strip():
        print(
            "refusing to launch without --save-folder / $EDULLM_CHECKPOINT_DIR "
            "(platform checkpoints would land on ephemeral local disk)",
            file=sys.stderr,
        )
        return 64

    cmd = [
        sys.executable,
        "-m",
        "scripts.train_olmo",
        "--config",
        str(args.config),
        "--save-folder",
        str(args.save_folder),
    ]
    if args.olmo_root is not None:
        cmd.extend(["--olmo-root", str(args.olmo_root)])
    if args.data_cache_dir is not None:
        cmd.extend(["--data-cache-dir", str(args.data_cache_dir)])
    if args.resume:
        cmd.append("--resume")
    if not args.dry_run:
        cmd.append("--launch")

    os.environ.setdefault("EDULLM_RUN_ID", str(args.run_name))
    os.execvp(cmd[0], cmd)
    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
