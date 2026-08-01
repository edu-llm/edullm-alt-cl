"""Three-phase data schedule for the HQ front-load experiment.

Phase A — LR warmup on general pretrain (``pretrain/regmix-10b``)
Phase B — HQ front-load (``pretrain/hq-frontload-100m``)
Phase C — remaining general pretrain (``pretrain/regmix-10b``)

Defaults match the shared OLMo2-370M / RegMix-10B contract:
GBS ``4_194_304`` tokens/step, LR warmup 24 steps ≈ 100M tokens,
total ≈ 10B tokens → 2384 steps. Phase B matches Phase A length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

PhaseName = Literal["A_warmup_regmix", "B_hq", "C_regmix", "control"]

GLOBAL_BATCH_TOKENS = 4_194_304
TOTAL_STEPS = 2384
PHASE_A_STEPS = 24  # ~100M tokens
PHASE_B_STEPS = 24  # ~100M tokens
PHASE_C_START = PHASE_A_STEPS + PHASE_B_STEPS  # 48


@dataclass(frozen=True)
class PhaseSpec:
    name: PhaseName
    start_step: int  # inclusive
    end_step: int  # exclusive
    corpus: Literal["regmix", "hq"]

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step

    @property
    def n_tokens(self) -> int:
        return self.n_steps * GLOBAL_BATCH_TOKENS


DEFAULT_PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("A_warmup_regmix", 0, PHASE_A_STEPS, "regmix"),
    PhaseSpec("B_hq", PHASE_A_STEPS, PHASE_C_START, "hq"),
    PhaseSpec("C_regmix", PHASE_C_START, TOTAL_STEPS, "regmix"),
)


def phase_boundaries(
    *,
    phase_a_steps: int = PHASE_A_STEPS,
    phase_b_steps: int = PHASE_B_STEPS,
    total_steps: int = TOTAL_STEPS,
) -> tuple[PhaseSpec, ...]:
    """Build phase specs; raises if budgets don't fit in ``total_steps``."""
    a = int(phase_a_steps)
    b = int(phase_b_steps)
    t = int(total_steps)
    if a < 0 or b < 0:
        raise ValueError(f"phase steps must be >= 0, got A={a} B={b}")
    if a + b > t:
        raise ValueError(
            f"phase A ({a}) + B ({b}) = {a + b} exceeds total_steps={t}"
        )
    return (
        PhaseSpec("A_warmup_regmix", 0, a, "regmix"),
        PhaseSpec("B_hq", a, a + b, "hq"),
        PhaseSpec("C_regmix", a + b, t, "regmix"),
    )


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
        # Post-final bookkeeping; clamp to last phase.
        return last
    raise ValueError(
        f"step {s} outside schedule [0, {last.end_step}]"
    )


def corpus_for_step(
    step: int,
    phases: Sequence[PhaseSpec] = DEFAULT_PHASES,
    *,
    control: bool = False,
) -> Literal["regmix", "hq"]:
    """Which token pool to sample for ``step``. Control always uses regmix."""
    if control:
        return "regmix"
    return phase_for_step(step, phases).corpus


def tokens_seen_after(step: int, *, gbs: int = GLOBAL_BATCH_TOKENS) -> int:
    """Tokens consumed after completing ``step`` (1-indexed completed count)."""
    return int(step) * int(gbs)
