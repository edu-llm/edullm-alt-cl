"""Unit tests for the HQ front-load phase schedule (no GPU / S3 required)."""

from __future__ import annotations

import pytest

from alt_cl.checkpoint_ladder import permanent_checkpoint_steps
from alt_cl.phases import (
    GLOBAL_BATCH_TOKENS,
    PHASE_A_STEPS,
    PHASE_B_STEPS,
    PHASE_C_START,
    TOTAL_STEPS,
    corpus_for_step,
    phase_boundaries,
    phase_for_step,
)


def test_default_boundaries_sum_to_total():
    phases = phase_boundaries()
    assert phases[0].start_step == 0
    assert phases[0].end_step == PHASE_A_STEPS
    assert phases[1].start_step == PHASE_A_STEPS
    assert phases[1].end_step == PHASE_C_START
    assert phases[2].end_step == TOTAL_STEPS
    assert sum(p.n_steps for p in phases) == TOTAL_STEPS


def test_phase_a_and_b_are_about_100m_tokens():
    phases = phase_boundaries()
    assert phases[0].n_tokens == PHASE_A_STEPS * GLOBAL_BATCH_TOKENS
    assert phases[1].n_tokens == PHASE_B_STEPS * GLOBAL_BATCH_TOKENS
    # ~100M each (24 * 4_194_304 = 100_663_296)
    assert abs(phases[0].n_tokens - 100_000_000) < 1_000_000
    assert abs(phases[1].n_tokens - 100_000_000) < 1_000_000


def test_phase_for_step_transitions():
    assert phase_for_step(0).name == "A_warmup_regmix"
    assert phase_for_step(23).name == "A_warmup_regmix"
    assert phase_for_step(24).name == "B_hq"
    assert phase_for_step(47).name == "B_hq"
    assert phase_for_step(48).name == "C_regmix"
    assert phase_for_step(2383).name == "C_regmix"


def test_corpus_switching_and_control():
    assert corpus_for_step(10) == "regmix"
    assert corpus_for_step(30) == "hq"
    assert corpus_for_step(100) == "regmix"
    assert corpus_for_step(30, control=True) == "regmix"


def test_phase_budget_validation():
    with pytest.raises(ValueError):
        phase_boundaries(phase_a_steps=2000, phase_b_steps=2000, total_steps=2384)


def test_checkpoint_ladder_2384_omits_near_final_grid():
    steps = permanent_checkpoint_steps(2384)
    assert 0 in steps and 2384 in steps
    assert 2250 in steps
    assert 2375 not in steps
