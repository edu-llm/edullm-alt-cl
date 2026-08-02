"""Validate an alt-cl YAML config without launching training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alt_cl.contract import (  # noqa: E402
    default_total_steps_for_config,
    load_config,
    run_fingerprint,
    validate_config,
    verify_olmo_revision,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--olmo-root",
        type=Path,
        default=None,
        help="If set, verify git HEAD matches olmo_core.revision",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        validate_config(cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.olmo_root is not None:
        rev = str((cfg.get("olmo_core") or {}).get("revision") or "")
        try:
            verify_olmo_revision(args.olmo_root, rev)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    fp = run_fingerprint(cfg)
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": cfg["run_id"],
                "arm": cfg["arm"],
                "total_steps": default_total_steps_for_config(cfg),
                "fingerprint": fp,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
