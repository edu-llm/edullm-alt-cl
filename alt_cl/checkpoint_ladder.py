"""Permanent checkpoint ladder (shared eduLLM 370M contract)."""

from __future__ import annotations

from typing import List

DEFAULT_CHECKPOINT_INTERVAL = 125


def permanent_checkpoint_steps(
    total_steps: int,
    interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> List[int]:
    """Sorted permanent save steps: 0, every ``interval``, true final.

    Omits the last on-grid step when it falls within one interval of the final
    (e.g. 2384-step run omits 2375).
    """
    if total_steps < 0:
        raise ValueError(f"total_steps must be >= 0, got {total_steps}")
    if interval <= 0:
        raise ValueError(f"interval must be > 0, got {interval}")
    if total_steps == 0:
        return [0]

    steps = {0, int(total_steps)}
    last_grid = (int(total_steps) // int(interval)) * int(interval)
    for s in range(int(interval), last_grid + 1, int(interval)):
        steps.add(s)
    if (
        last_grid > 0
        and last_grid != int(total_steps)
        and (int(total_steps) - last_grid) < int(interval)
    ):
        steps.discard(last_grid)
    return sorted(steps)
