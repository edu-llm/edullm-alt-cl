"""Deterministic FineMath doc ranking for reproducible frontload samples."""

from __future__ import annotations

import hashlib
from typing import Any


def rank_key(seed: int, url: str, warc_record_offset: Any) -> str:
    """Ascending sha256 hex key: ``f\"{seed}:{url}:{warc_record_offset}\"``."""
    raw = f"{int(seed)}:{url}:{warc_record_offset}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
