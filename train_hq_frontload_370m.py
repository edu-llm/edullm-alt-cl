#!/usr/bin/env python3
"""Deprecated entrypoint — use ``scripts.train_olmo`` with a YAML config.

The hand-rolled train loop has been replaced by an OLMo-core ``Trainer`` path.
See README.md and ``configs/``.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "train_hq_frontload_370m.py is deprecated.\n"
        "Use:\n"
        "  python -m scripts.train_olmo --config configs/math_front_anneal_10b.yaml\n"
        "  # or launch:\n"
        "  OLMO_ROOT=/path/to/OLMo-core CONFIG=configs/math_front_anneal_10b.yaml "
        "./launch/launch_arm.sh\n"
    )


if __name__ == "__main__":
    main()
    sys.exit(1)
