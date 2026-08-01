"""Resolve + stage published corpora from ``s3://edullm-data``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

log = logging.getLogger("alt_cl.data")

DATA_BUCKET = "edullm-data"
LEGACY_DATA_BUCKET = "edullm-datasets"

DEFAULT_REGMIX_DATASET_ID = "pretrain/regmix-10b"
DEFAULT_HQ_DATASET_ID = "pretrain/hq-frontload-100m"


def default_data_cache_dir() -> Path:
    for key in ("EDULLM_DATA_CACHE", "SCRATCH", "TMPDIR"):
        val = os.environ.get(key)
        if val:
            return Path(val) / "edullm-data-cache"
    return Path.cwd() / "edullm-data-cache"


def _refuse_legacy_uri(uri: str) -> None:
    if LEGACY_DATA_BUCKET in uri:
        raise SystemExit(
            f"refusing legacy training URI (use s3://{DATA_BUCKET}/ via edullm_data): {uri}"
        )


def _parse_edullm_data_uri(uri: str) -> Tuple[str, str]:
    _refuse_legacy_uri(uri)
    if not uri.startswith("s3://"):
        raise SystemExit(f"expected s3:// URI from dataset_paths, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if bucket != DATA_BUCKET:
        raise SystemExit(
            f"only s3://{DATA_BUCKET}/ training URIs are allowed; got {uri}"
        )
    if not key:
        raise SystemExit(f"empty key in URI {uri}")
    return bucket, key


def _edullm_s3():
    try:
        from edullm_data.s3 import Boto3S3
    except ImportError as e:
        raise SystemExit(
            "edullm-data package is required "
            '(pip/uv: "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.6.3")'
        ) from e
    return Boto3S3.default()


def resolve_published_split(
    dataset_id: str,
    *,
    version: Optional[str] = None,
    split: str = "train",
):
    try:
        from edullm_data.read import dataset_paths, resolve_latest
    except ImportError as e:
        raise SystemExit(
            "edullm-data package is required "
            '(pip/uv: "edullm-data @ git+https://github.com/edu-llm/edullm-data@v0.6.3")'
        ) from e

    s3 = _edullm_s3()
    ver = version or resolve_latest(dataset_id, s3=s3)
    if not ver:
        raise SystemExit(
            f"no published version of {dataset_id!r} under "
            f"s3://{DATA_BUCKET}/_catalog/ — publish+validate before training "
            f"(see DATASET-DESIGN.md for {DEFAULT_HQ_DATASET_ID})"
        )
    resolved = dataset_paths(dataset_id, ver, split=split, s3=s3)
    if not resolved.paths:
        raise SystemExit(
            f"{dataset_id}/{ver} split={split!r} resolved to zero paths"
        )
    for uri in resolved.paths:
        _parse_edullm_data_uri(uri)
    return resolved, ver


def stage_edullm_uris(uris: List[str], cache_dir: Path) -> List[str]:
    import boto3

    client = boto3.client("s3", region_name="us-east-1")
    local_paths: List[str] = []
    cache_dir = Path(cache_dir)
    for uri in uris:
        bucket, key = _parse_edullm_data_uri(uri)
        dest = cache_dir / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        need_fetch = True
        if dest.is_file():
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                if int(dest.stat().st_size) == int(head["ContentLength"]):
                    need_fetch = False
            except Exception as e:  # noqa: BLE001
                log.warning("HEAD %s failed (%s); re-fetching", uri, e)
        if need_fetch:
            log.info("staging %s → %s", uri, dest)
            tmp = dest.with_name(dest.name + ".partial")
            client.download_file(bucket, key, str(tmp))
            tmp.replace(dest)
        local_paths.append(str(dest))
    return local_paths


def resolve_and_stage_train_tokens(
    *,
    dataset_id: str,
    version: Optional[str],
    cache_dir: Path,
) -> Tuple[List[str], str, Any]:
    """Return (local_paths, resolved_version, numpy_dtype)."""
    resolved, ver = resolve_published_split(
        dataset_id, version=version, split="train"
    )
    dtype = resolved.numpy_dtype or "uint32"
    if getattr(resolved, "header_bytes", None):
        raise SystemExit(
            f"{dataset_id}/{ver} declares header_bytes={resolved.header_bytes}; "
            "this trainer only memmaps headerless .u32le.bin shards"
        )
    local = stage_edullm_uris(list(resolved.paths), cache_dir)
    return local, ver, dtype


def write_stage_meta(
    path: Path,
    *,
    dataset_id: str,
    version: str,
    paths: List[str],
    dtype: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "version": version,
                "paths": paths,
                "dtype": str(dtype),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_paths_file(path: Path) -> List[str]:
    paths = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not paths:
        raise SystemExit(f"No paths in {path}")
    return paths
