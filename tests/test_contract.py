"""Unit tests for alt-cl config contract + resume fingerprints."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alt_cl.contract import (
    compare_fingerprints,
    load_config,
    run_fingerprint,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.mark.parametrize(
    "name",
    ["math_front_anneal_10b.yaml", "math_anneal_10b.yaml", "smoke.yaml"],
)
def test_shipped_configs_validate(name: str):
    cfg = load_config(CONFIGS / name)
    validate_config(cfg)
    fp = run_fingerprint(cfg)
    assert fp["run_id"] == cfg["run_id"]
    assert fp["arm"] == cfg["arm"]
    assert fp["olmo_revision"]
    assert fp["math_dataset_id"] == "pretrain/math-frontload-100m"
    train = cfg.get("train") or {}
    assert train.get("ephemeral_checkpoint_every_steps") is None
    assert train.get("checkpoint_keep_last") is None or name == "smoke.yaml"


def test_fingerprint_allows_max_tokens_extend_only():
    cfg = load_config(CONFIGS / "math_anneal_10b.yaml")
    prior = run_fingerprint(cfg)
    current = dict(prior)
    current["max_tokens"] = int(prior["max_tokens"]) + 4_194_304
    assert compare_fingerprints(prior, current) is None
    current["seed"] = int(prior["seed"]) + 1
    err = compare_fingerprints(prior, current)
    assert err is not None
    assert "seed" in err or "differing" in err


def test_fingerprint_rejects_arm_change():
    cfg = load_config(CONFIGS / "math_anneal_10b.yaml")
    prior = run_fingerprint(cfg)
    current = dict(prior)
    current["arm"] = "math-front-anneal"
    assert compare_fingerprints(prior, current) is not None


def test_validate_rejects_missing_revision(tmp_path: Path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg = dict(cfg)
    cfg["olmo_core"] = dict(cfg.get("olmo_core") or {})
    cfg["olmo_core"]["revision"] = ""
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    bad = load_config(path)
    with pytest.raises(ValueError, match="revision"):
        validate_config(bad)


def test_validate_rejects_unknown_arm(tmp_path: Path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg = dict(cfg)
    cfg["arm"] = "rel-ema"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="arm"):
        validate_config(load_config(path))


def test_validate_rejects_max_tokens_not_multiple_of_gbs(tmp_path: Path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    cfg = dict(cfg)
    train = dict(cfg.get("train") or {})
    train["max_tokens"] = int(train["global_batch_size"]) * 12 + 1
    cfg["train"] = train
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="exact multiple"):
        validate_config(load_config(path))


def test_resolve_output_dir_env_override(monkeypatch, tmp_path: Path):
    from alt_cl.contract import resolve_output_dir

    cfg = {"output_dir": "from-yaml"}
    monkeypatch.delenv("SAVE_FOLDER", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    monkeypatch.delenv("EDULLM_CHECKPOINT_DIR", raising=False)
    assert resolve_output_dir(cfg, tmp_path) == tmp_path / "from-yaml"
    monkeypatch.setenv("SAVE_FOLDER", str(tmp_path / "from-env"))
    assert resolve_output_dir(cfg, tmp_path) == tmp_path / "from-env"
    # Remote SAVE_FOLDER must not become the local scratch root.
    monkeypatch.setenv("SAVE_FOLDER", "s3://bucket/teams/t/runs/r/checkpoints")
    assert resolve_output_dir(cfg, tmp_path) == tmp_path / "from-yaml"


def test_resolve_save_folder_precedence(monkeypatch, tmp_path: Path):
    from alt_cl.platform_env import is_remote_uri, resolve_save_folder

    assert is_remote_uri("s3://bucket/key")
    assert not is_remote_uri(tmp_path / "local")

    cfg = {"output_dir": "out"}
    monkeypatch.delenv("EDULLM_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("SAVE_FOLDER", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    assert resolve_save_folder(cfg, tmp_path) == str(tmp_path / "out" / "checkpoints")

    monkeypatch.setenv(
        "EDULLM_CHECKPOINT_DIR",
        "s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/run_x/checkpoints/",
    )
    assert resolve_save_folder(cfg, tmp_path).startswith("s3://")
    assert (
        resolve_save_folder(cfg, tmp_path, cli_save_folder="s3://cli/checkpoints")
        == "s3://cli/checkpoints"
    )


def test_s3_checkpoint_uri_prefers_env(monkeypatch):
    from alt_cl.contract import s3_checkpoint_uri

    cfg = {
        "s3": {
            "checkpoint_bucket": "ignored",
            "prefix": "ignored",
        }
    }
    monkeypatch.setenv(
        "EDULLM_CHECKPOINT_DIR",
        "s3://sbsandbox-intern-edullm-outputs/teams/t/runs/r/checkpoints",
    )
    assert s3_checkpoint_uri(cfg) == (
        "s3://sbsandbox-intern-edullm-outputs/teams/t/runs/r/checkpoints"
    )
