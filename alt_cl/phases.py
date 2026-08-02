"""Phase schedules for the math front-load + end anneal experiment.

Both arms share the same end-of-pretrain **math anneal** (~100M math tokens
woven into the last ~500M tokens, linear mix 0 → ~40%). Treatment also gets an
exclusive ~100M math block right after LR warmup.

* ``math-front-anneal`` (treatment): warmup → exclusive math → bulk regmix → anneal
* ``math-anneal`` (control):         warmup → bulk regmix → same anneal

LR warmup is always the first 24 regmix steps; after that LR stays constant
(including through exclusive math and the anneal), so early math is not
confounded with LR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ArmName = Literal["math-front-anneal", "math-anneal"]
PhaseName = Literal["A_warmup_regmix", "B_front_math", "C_regmix", "D_anneal"]
CorpusName = Literal["regmix", "math", "mix"]

GLOBAL_BATCH_TOKENS = 4_194_304
REGMIX_STEPS = 2384  # full normal ~10B regmix-scheduled run
FRONT_MATH_STEPS = 24  # ~100M exclusive math (treatment only)
ANNEAL_MATH_STEPS = 24  # ~100M math *mass* inside the anneal window (both arms)
# Last ~500M tokens: integral of linear 0→p_max equals ANNEAL_MATH_STEPS
# (mean mix ~20%, peak ~40%)
ANNEAL_WINDOW_STEPS = 119  # 119 * GBS ≈ 499.1M ≈ 500M
WARMUP_STEPS = 24  # LR warmup on regmix
# Treatment adds exclusive front math on top of the 10B schedule
TOTAL_STEPS_CONTROL = REGMIX_STEPS  # 2384
TOTAL_STEPS_TREATMENT = REGMIX_STEPS + FRONT_MATH_STEPS  # 2408

# Back-compat aliases
HQ_STEPS = FRONT_MATH_STEPS
TOTAL_STEPS = TOTAL_STEPS_TREATMENT
PHASE_A_STEPS = WARMUP_STEPS
PHASE_B_STEPS = FRONT_MATH_STEPS
TOTAL_REGMIX_STEPS = REGMIX_STEPS


def anneal_p_max(
    *,
    anneal_math_steps: int = ANNEAL_MATH_STEPS,
    anneal_window_steps: int = ANNEAL_WINDOW_STEPS,
) -> float:
    """Peak math fraction for a linear 0→p_max ramp whose mean is math/window."""
    w = int(anneal_window_steps)
    m = int(anneal_math_steps)
    if w <= 0:
        raise ValueError(f"anneal_window_steps must be > 0, got {w}")
    if m < 0 or m > w:
        raise ValueError(f"anneal_math_steps ({m}) must be in [0, {w}]")
    # mean_p = m/w = p_max/2  →  p_max = 2m/w
    return (2.0 * m) / w


@dataclass(frozen=True)
class PhaseSpec:
    name: PhaseName
    start_step: int  # inclusive
    end_step: int  # exclusive
    corpus: CorpusName
    math_frac_start: float = 0.0
    math_frac_end: float = 0.0

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step

    @property
    def n_tokens(self) -> int:
        return self.n_steps * GLOBAL_BATCH_TOKENS


def phase_boundaries(
    arm: ArmName = "math-front-anneal",
    *,
    warmup_steps: int = WARMUP_STEPS,
    front_math_steps: int = FRONT_MATH_STEPS,
    regmix_steps: int = REGMIX_STEPS,
    anneal_window_steps: int = ANNEAL_WINDOW_STEPS,
    anneal_math_steps: int = ANNEAL_MATH_STEPS,
) -> tuple[PhaseSpec, ...]:
    """Build phase specs for ``arm``.

    The anneal occupies the last ``anneal_window_steps`` of the regmix-scheduled
    timeline (shared shape for both arms). Exclusive front math is additive and
    only present on ``math-front-anneal``.
    """
    a = int(warmup_steps)
    f = int(front_math_steps)
    r = int(regmix_steps)
    w = int(anneal_window_steps)
    m = int(anneal_math_steps)
    if min(a, f, r, w) < 0 or m < 0:
        raise ValueError(
            f"steps must be >= 0, got warmup={a} front={f} regmix={r} "
            f"anneal_window={w} anneal_math={m}"
        )
    if a > r:
        raise ValueError(f"warmup_steps ({a}) cannot exceed regmix_steps ({r})")
    if w > r - a:
        raise ValueError(
            f"anneal_window_steps ({w}) leaves no bulk regmix after warmup "
            f"(regmix={r}, warmup={a})"
        )
    if m > w:
        raise ValueError(f"anneal_math_steps ({m}) cannot exceed anneal_window ({w})")

    p_max = anneal_p_max(anneal_math_steps=m, anneal_window_steps=w)
    name = str(arm).strip().lower().replace("_", "-")

    # Legacy aliases from the hard-block design
    if name in ("hq-frontload", "math-frontload"):
        name = "math-front-anneal"
    if name in ("hq-backload", "math-backload"):
        name = "math-anneal"

    if name == "math-front-anneal":
        # [0,a) warmup → [a,a+f) exclusive math → bulk regmix → anneal
        # Anneal sits in the last `w` steps of the *regmix-scheduled* timeline,
        # shifted by +f because front math is additive.
        bulk_end = f + (r - w)  # exclusive end of pure-regmix bulk
        total = f + r
        return (
            PhaseSpec("A_warmup_regmix", 0, a, "regmix"),
            PhaseSpec("B_front_math", a, a + f, "math", 1.0, 1.0),
            PhaseSpec("C_regmix", a + f, bulk_end, "regmix"),
            PhaseSpec("D_anneal", bulk_end, total, "mix", 0.0, p_max),
        )
    if name == "math-anneal":
        bulk_end = r - w
        return (
            PhaseSpec("A_warmup_regmix", 0, a, "regmix"),
            PhaseSpec("C_regmix", a, bulk_end, "regmix"),
            PhaseSpec("D_anneal", bulk_end, r, "mix", 0.0, p_max),
        )
    raise ValueError(
        f"unknown arm {arm!r}; expected math-front-anneal|math-anneal"
    )


DEFAULT_PHASES: tuple[PhaseSpec, ...] = phase_boundaries("math-front-anneal")


def phase_for_step(
    step: int,
    phases: Sequence[PhaseSpec] = DEFAULT_PHASES,
) -> PhaseSpec:
    """Return the phase active while *taking* training step ``step`` (0-indexed)."""
    s = int(step)
    if s < 0:
        raise ValueError(f"step must be >= 0, got {s}")
    for p in phases:
        if p.start_step <= s < p.end_step:
            return p
    last = phases[-1]
    if s == last.end_step:
        return last
    raise ValueError(f"step {s} outside schedule [0, {last.end_step}]")


def math_fraction_for_step(
    step: int,
    phases: Sequence[PhaseSpec] = DEFAULT_PHASES,
) -> float:
    """Math token fraction for this step (0=regmix, 1=exclusive math, else anneal)."""
    p = phase_for_step(step, phases)
    if p.corpus == "math":
        return 1.0
    if p.corpus == "regmix":
        return 0.0
    n = p.n_steps
    if n <= 1:
        return float(p.math_frac_end)
    # Left-closed: first anneal step at frac_start, last at frac_end
    t = (int(step) - p.start_step) / (n - 1)
    return float(p.math_frac_start + t * (p.math_frac_end - p.math_frac_start))


def corpus_for_step(
    step: int,
    phases: Sequence[PhaseSpec] = DEFAULT_PHASES,
) -> CorpusName:
    """Dominant corpus label for logging; prefer ``math_fraction_for_step`` for mixing."""
    return phase_for_step(step, phases).corpus


def tokens_seen_after(step: int, *, gbs: int = GLOBAL_BATCH_TOKENS) -> int:
    """Tokens consumed after completing ``step`` (1-indexed completed count)."""
    return int(step) * int(gbs)


def total_steps_for_arm(arm: ArmName) -> int:
    name = str(arm).strip().lower().replace("_", "-")
    if name in ("hq-frontload", "math-frontload", "math-front-anneal"):
        return TOTAL_STEPS_TREATMENT
    if name in ("hq-backload", "math-backload", "math-anneal"):
        return TOTAL_STEPS_CONTROL
    raise ValueError(f"unknown arm {arm!r}")
