"""Tests for platform path helpers."""

from __future__ import annotations

from pathlib import Path

from alt_cl.platform_env import (
    is_remote_uri,
    join_uri,
    resolve_run_scratch,
    resolve_save_folder,
)


def test_join_uri():
    assert join_uri("s3://b/p/", "run_fingerprint.json") == (
        "s3://b/p/run_fingerprint.json"
    )


def test_resolve_run_scratch_local(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("EDULLM_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("SAVE_FOLDER", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("EDULLM_RUN_ID", raising=False)
    cfg = {"output_dir": "run", "run_id": "local-run"}
    save = resolve_save_folder(cfg, tmp_path)
    scratch = resolve_run_scratch(cfg, tmp_path, save_folder=save)
    assert scratch["metrics_dir"].endswith("metrics")
    assert Path(scratch["output_dir"]) == tmp_path / "run"


def test_resolve_run_scratch_remote_uses_tmp(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDULLM_RUN_ID", "run_test123")
    monkeypatch.delenv("SAVE_FOLDER", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    cfg = {"output_dir": "run", "run_id": "yaml-id"}
    save = "s3://sbsandbox-intern-edullm-outputs/teams/t/runs/run_test123/checkpoints"
    assert is_remote_uri(save)
    scratch = resolve_run_scratch(cfg, tmp_path, save_folder=save)
    assert "edullm-alt-cl" in scratch["work_dir"]
    assert "run_test123" in scratch["work_dir"]
