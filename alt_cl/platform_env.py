"""eduLLM platform path / env resolution for alt-cl runs.

Platform injects:

  EDULLM_RUN_ID
  EDULLM_CHECKPOINT_DIR   s3://…/teams/{team}/runs/{run_id}/checkpoints/
  EDULLM_OUTPUT_PREFIX    s3://…/teams/{team}/runs/{run_id}/

Pretrain resolves both corpora from YAML (form ``dataset_release: none``).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional


def is_remote_uri(path: str | Path | None) -> bool:
    return str(path or "").strip().lower().startswith("s3://")


def platform_run_id() -> Optional[str]:
    val = (os.environ.get("EDULLM_RUN_ID") or "").strip()
    return val or None


def _env_uri(*keys: str) -> Optional[str]:
    for key in keys:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def resolve_output_dir(cfg: Mapping[str, Any], root: Path) -> Path:
    """Local scratch root; ``SAVE_FOLDER`` / ``OUTPUT_DIR`` win over YAML ``output_dir``."""
    env = _env_uri("SAVE_FOLDER", "OUTPUT_DIR")
    # SAVE_FOLDER may be an s3:// checkpoint dir on platform — do not use as local root.
    if env and is_remote_uri(env):
        env = None
    raw = env or str(cfg.get("output_dir") or "").strip()
    if not raw:
        raise ValueError("output_dir is required (or set SAVE_FOLDER / OUTPUT_DIR)")
    out = Path(raw)
    if not out.is_absolute():
        out = root / out
    return out


def resolve_save_folder(
    cfg: Mapping[str, Any],
    root: Path,
    *,
    cli_save_folder: Optional[str] = None,
) -> str:
    """Checkpoint dir: CLI > EDULLM_CHECKPOINT_DIR > SAVE_FOLDER (if s3) > output_dir/checkpoints."""
    cli = (cli_save_folder or "").strip()
    if cli:
        return cli
    env_ckpt = _env_uri("EDULLM_CHECKPOINT_DIR")
    if env_ckpt:
        return env_ckpt
    save_folder_env = (os.environ.get("SAVE_FOLDER") or "").strip()
    if save_folder_env and is_remote_uri(save_folder_env):
        return save_folder_env
    out = resolve_output_dir(cfg, root)
    return str(out / "checkpoints")


def resolve_run_scratch(
    cfg: Mapping[str, Any],
    root: Path,
    *,
    save_folder: str,
) -> dict[str, str]:
    """Local metrics / work / cache dirs (never under an s3:// checkpoint prefix)."""
    out = resolve_output_dir(cfg, root)
    # EDULLM_OUTPUT_PREFIX is s3:// — keep heavy local work on disk; metrics may be local too.
    # Use a dedicated local scratch when the only configured out would collide with remote.
    if is_remote_uri(save_folder) or is_remote_uri(str(out)):
        run = platform_run_id() or str(cfg.get("run_id") or "local")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run)[:80]
        scratch = Path(tempfile.gettempdir()) / "edullm-alt-cl" / safe
    else:
        scratch = out
    return {
        "metrics_dir": str(scratch / "metrics"),
        "progress_dir": str(scratch / "progress"),
        "dataset_cache": str(scratch / "dataset_cache"),
        "work_dir": str(scratch / "dataloader_work"),
        "output_dir": str(scratch),
    }


def ensure_local_dir(path: str | Path) -> Path:
    p = Path(path)
    if is_remote_uri(p):
        raise ValueError(f"refusing to mkdir remote URI as local dir: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    return bucket, key


def join_uri(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name.lstrip('/')}"


def read_json_uri(uri: str) -> Optional[dict[str, Any]]:
    if is_remote_uri(uri):
        import boto3

        bucket, key = _parse_s3_uri(uri)
        client = boto3.client("s3", region_name="us-east-1")
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except client.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return json.loads(obj["Body"].read().decode("utf-8"))
    p = Path(uri)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_json_uri(uri: str, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), indent=2) + "\n"
    if is_remote_uri(uri):
        import boto3

        bucket, key = _parse_s3_uri(uri)
        client = boto3.client("s3", region_name="us-east-1")
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return
    p = Path(uri)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def local_dir_nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())
