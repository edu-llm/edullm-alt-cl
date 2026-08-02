"""Unit tests for math front-load + end anneal phase schedules."""

from __future__ import annotations

import pytest

from alt_cl.checkpoint_ladder import permanent_checkpoint_steps
from alt_cl.phases import (
    ANNEAL_MATH_STEPS,
    ANNEAL_WINDOW_STEPS,
    FRONT_MATH_STEPS,
    GLOBAL_BATCH_TOKENS,
    REGMIX_STEPS,
    TOTAL_STEPS_CONTROL,
    TOTAL_STEPS_TREATMENT,
    WARMUP_STEPS,
    anneal_p_max,
    corpus_for_step,
    math_fraction_for_step,
    phase_boundaries,
    phase_for_step,
    total_steps_for_arm,
)


def test_anneal_p_max_gives_100m_math_mass():
    p_max = anneal_p_max()
    # linear 0→p_max over window → mean p_max/2 → math steps = window * p_max/2
    assert ANNEAL_WINDOW_STEPS * p_max / 2 == pytest.approx(ANNEAL_MATH_STEPS)
    assert p_max == pytest.approx(2 * ANNEAL_MATH_STEPS / ANNEAL_WINDOW_STEPS)
    assert 0.39 < p_max < 0.41  # ~40% peak (mean ~20% over ~500M)


def test_front_and_anneal_math_mass_about_100m_each():
    assert FRONT_MATH_STEPS * GLOBAL_BATCH_TOKENS == pytest.approx(100_663_296)
    assert ANNEAL_MATH_STEPS * GLOBAL_BATCH_TOKENS == pytest.approx(100_663_296)
    assert abs(ANNEAL_WINDOW_STEPS * GLOBAL_BATCH_TOKENS - 500_000_000) < 5_000_000


def test_both_arms_share_anneal_shape():
    treat = phase_boundaries("math-front-anneal")
    ctrl = phase_boundaries("math-anneal")
    t_anneal = [p for p in treat if p.name == "D_anneal"][0]
    c_anneal = [p for p in ctrl if p.name == "D_anneal"][0]
    assert t_anneal.n_steps == c_anneal.n_steps == ANNEAL_WINDOW_STEPS
    assert t_anneal.math_frac_start == c_anneal.math_frac_start == 0.0
    assert t_anneal.math_frac_end == pytest.approx(c_anneal.math_frac_end)
    assert t_anneal.math_frac_end == pytest.approx(anneal_p_max())


def test_treatment_has_exclusive_front_control_does_not():
    treat = phase_boundaries("math-front-anneal")
    ctrl = phase_boundaries("math-anneal")
    assert any(p.name == "B_front_math" for p in treat)
    assert not any(p.name == "B_front_math" for p in ctrl)
    assert sum(p.n_steps for p in treat) == TOTAL_STEPS_TREATMENT
    assert sum(p.n_steps for p in ctrl) == TOTAL_STEPS_CONTROL
    assert TOTAL_STEPS_TREATMENT == REGMIX_STEPS + FRONT_MATH_STEPS
    assert TOTAL_STEPS_CONTROL == REGMIX_STEPS


def test_front_math_right_after_warmup():
    phases = phase_boundaries("math-front-anneal")
    assert phase_for_step(0, phases).name == "A_warmup_regmix"
    assert math_fraction_for_step(WARMUP_STEPS - 1, phases) == 0.0
    assert phase_for_step(WARMUP_STEPS, phases).name == "B_front_math"
    assert math_fraction_for_step(WARMUP_STEPS, phases) == 1.0
    assert corpus_for_step(WARMUP_STEPS, phases) == "math"


def test_control_has_no_math_until_anneal():
    phases = phase_boundaries("math-anneal")
    assert math_fraction_for_step(WARMUP_STEPS, phases) == 0.0
    anneal_start = REGMIX_STEPS - ANNEAL_WINDOW_STEPS
    assert math_fraction_for_step(anneal_start - 1, phases) == 0.0
    assert phase_for_step(anneal_start, phases).name == "D_anneal"
    assert math_fraction_for_step(anneal_start, phases) == pytest.approx(0.0)
    assert math_fraction_for_step(REGMIX_STEPS - 1, phases) == pytest.approx(anneal_p_max())


def test_anneal_fraction_ramps_linearly_on_treatment():
    phases = phase_boundaries("math-front-anneal")
    anneal = [p for p in phases if p.name == "D_anneal"][0]
    mid = anneal.start_step + anneal.n_steps // 2
    frac_mid = math_fraction_for_step(mid, phases)
    assert 0.0 < frac_mid < anneal.math_frac_end
    assert frac_mid == pytest.approx(
        anneal.math_frac_start
        + ((mid - anneal.start_step) / (anneal.n_steps - 1))
        * (anneal.math_frac_end - anneal.math_frac_start)
    )


def test_legacy_arm_aliases():
    assert total_steps_for_arm("hq-frontload") == TOTAL_STEPS_TREATMENT  # type: ignore[arg-type]
    assert total_steps_for_arm("hq-backload") == TOTAL_STEPS_CONTROL  # type: ignore[arg-type]
    front = phase_boundaries("hq-frontload")  # type: ignore[arg-type]
    assert any(p.name == "B_front_math" for p in front)


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        phase_boundaries("control")  # type: ignore[arg-type]


def test_checkpoint_ladder_includes_final_total():
    steps = permanent_checkpoint_steps(TOTAL_STEPS_TREATMENT)
    assert 0 in steps and TOTAL_STEPS_TREATMENT in steps
