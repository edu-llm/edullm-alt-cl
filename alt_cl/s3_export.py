"""Optional ``aws s3 sync`` for durable checkpoint / progress export."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger("alt_cl.s3_export")

_FALSEY = frozenset({"0", "false", "no", "off"})


def s3_export_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    if os.environ.get("S3_EXPORT", "1").strip().lower() in _FALSEY:
        return False
    if os.environ.get("SKIP_S3_UPLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return True


def sync_to_s3(
    local: Union[str, Path],
    remote: str,
    *,
    enabled: Optional[bool] = None,
    fail_closed: bool = False,
) -> bool:
    """Sync ``local`` → ``remote``. Returns True on success."""
    if not s3_export_enabled(enabled):
        log.info("S3 export disabled; skip %s", remote)
        return False
    if shutil.which("aws") is None:
        msg = f"aws CLI not on PATH; cannot export to {remote}"
        if fail_closed:
            raise RuntimeError(msg)
        log.warning("%s", msg)
        return False
    local_p = Path(local)
    if not local_p.exists():
        msg = f"S3 export skip: missing local path {local_p}"
        if fail_closed:
            raise RuntimeError(msg)
        log.warning("%s", msg)
        return False
    remote = remote if remote.endswith("/") else remote + "/"
    cmd = ["aws", "s3", "sync", str(local_p), remote, "--only-show-errors"]
    try:
        log.info("S3 export: %s", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as exc:  # noqa: BLE001
        if fail_closed:
            raise RuntimeError(f"durable S3 export failed: {local_p} → {remote}") from exc
        log.warning("S3 export failed (%s → %s): %s", local_p, remote, exc)
        return False
