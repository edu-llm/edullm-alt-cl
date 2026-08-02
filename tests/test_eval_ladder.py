"""Unit tests for the GSM8K / MATH eval ladder planner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alt_cl.contract import load_config
from alt_cl.eval_ladder import (
    PRIMARY_TASK,
    build_eval_jobs,
    build_eval_plan,
    build_olmo_eval_argv,
    checkpoint_dirname,
    join_checkpoint,
    sft_eval_steps,
    validate_eval_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval_math_ladder.yaml"
HARNESS = ROOT / "configs" / "eval_harness_olmo_core.yaml"


def test_shipped_eval_config_validates():
    cfg = load_config(CONFIG)
    validate_eval_config(cfg)
    assert PRIMARY_TASK in cfg["tasks"]
    assert cfg["harness"]["provider_kind"] == "olmo_core"
    assert cfg["harness"]["revision"]
    assert HARNESS.is_file()


def test_harness_yaml_sets_olmo_core_kind():
    data = yaml.safe_load(HARNESS.read_text(encoding="utf-8"))
    assert data["provider"]["kind"] == "olmo_core"
    assert "dolma2" in data["provider"]["tokenizer"]


def test_sft_eval_steps_matches_sft_config_ladder():
    # 228 steps / every 25 → omits near-final grid 225 (same as permanent_checkpoint_steps)
    steps = sft_eval_steps(228, 25, include_zero=False)
    assert steps[0] == 25
    assert steps[-1] == 228
    assert 225 not in steps
    assert 0 not in steps
    assert sft_eval_steps(228, 25, include_zero=True)[0] == 0


def test_join_checkpoint_local_and_s3():
    assert checkpoint_dirname(228) == "step228"
    assert join_checkpoint("/tmp/ckpts", 50) == str(Path("/tmp/ckpts") / "step50")
    assert (
        join_checkpoint("s3://bucket/runs/r/checkpoints", 25)
        == "s3://bucket/runs/r/checkpoints/step25"
    )


def test_build_eval_jobs_base_and_sft():
    cfg = load_config(CONFIG)
    jobs = build_eval_jobs(
        cfg,
        arm="math-front-anneal",
        checkpoint_dir="s3://b/sft/checkpoints",
        base_checkpoint="s3://b/pretrain/checkpoints/step2408",
        steps=[25, 228],
    )
    assert jobs[0].stage == "base"
    assert jobs[0].step is None
    assert jobs[0].checkpoint.endswith("step2408")
    assert [j.step for j in jobs[1:]] == [25, 228]
    assert all(PRIMARY_TASK in j.tasks for j in jobs)


def test_include_optional_appends_gsm_symbolic():
    cfg = load_config(CONFIG)
    jobs = build_eval_jobs(
        cfg,
        arm="math-anneal",
        checkpoint_dir="/tmp/ckpts",
        steps=[228],
        include_optional=True,
    )
    assert "gsm_symbolic" in jobs[0].tasks
    assert jobs[0].tasks[0] == "gsm8k"


def test_rejects_unknown_arm_and_missing_gsm8k(tmp_path: Path):
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match="arm"):
        build_eval_jobs(
            cfg,
            arm="rel-ema",
            checkpoint_dir="/tmp/x",
            steps=[228],
        )
    bad = dict(cfg)
    bad["tasks"] = ["minerva_math"]
    with pytest.raises(ValueError, match="gsm8k"):
        validate_eval_config(bad)


def test_plan_commands_use_harness_and_dry_run():
    cfg = load_config(CONFIG)
    plan = build_eval_plan(
        cfg,
        root=ROOT,
        arm="math-anneal",
        checkpoint_dir="s3://b/ckpts",
        output_dir="s3://b/out/eval",
        base_checkpoint="s3://b/base/step2384",
        steps=[228],
        dry_run=True,
    )
    assert plan["harness_revision"]
    assert plan["provider_kind"] == "olmo_core"
    assert len(plan["jobs"]) == 2
    cmd = plan["commands"][0]
    assert cmd[0] == "olmo-eval"
    assert "run" in cmd
    assert "--harness-config" in cmd
    assert "--dry-run" in cmd
    assert "-t" in cmd and "gsm8k" in cmd
    launch = build_olmo_eval_argv(
        build_eval_jobs(
            cfg,
            arm="math-anneal",
            checkpoint_dir="s3://b/ckpts",
            steps=[228],
        )[0],
        harness_config=HARNESS,
        output_dir="/tmp/out",
        dry_run=False,
    )
    assert "--dry-run" not in launch
