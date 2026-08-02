"""OLMo-core ``TextDataLoaderBase`` that applies the alt-cl phase mix schedule."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from alt_cl.phases import PhaseSpec, math_fraction_for_step, phase_for_step
from alt_cl.streams import InfiniteBatchStream, MemmapTokenDataset, next_rank_input_ids

try:  # pragma: no cover - OLMo-core is cluster-only for most local CI.
    from olmo_core.data.collator import DataCollator
    from olmo_core.data.data_loader import TextDataLoaderBase

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    DataCollator = object  # type: ignore
    TextDataLoaderBase = object  # type: ignore
    _HAS_OLMO = False


def has_olmo_core() -> bool:
    return bool(_HAS_OLMO)


class CurriculumDataLoader(TextDataLoaderBase):  # type: ignore[misc]
    """Per-step regmix/math mix driven by ``alt_cl.phases``.

    Each optimizer step is one global batch. Within a step the math fraction is
    ``math_fraction_for_step`` and sequences are taken from the math stream via
    ``round(frac * seqs_per_rank)`` (rest from regmix), matching the locked
    experiment contract.
    """

    def __init__(
        self,
        *,
        regmix_paths: Sequence[str],
        math_paths: Optional[Sequence[str]],
        phases: Sequence[PhaseSpec],
        sequence_length: int,
        global_batch_size: int,
        work_dir: str | Path,
        seed: int,
        collator: Any,
        num_workers: int = 0,
        regmix_dtype: Any = np.uint32,
        math_dtype: Any = np.uint32,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        if not _HAS_OLMO:
            raise ImportError(
                "olmo_core is required for CurriculumDataLoader "
                "(pip install -e /path/to/edu-llm/OLMo-core)"
            )
        super().__init__(
            collator=collator,
            work_dir=work_dir,
            global_batch_size=int(global_batch_size),
            dp_world_size=int(dp_world_size),
            dp_rank=int(dp_rank),
            fs_local_rank=fs_local_rank,
        )
        self.phases = tuple(phases)
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        if self.global_batch_size % self.sequence_length != 0:
            raise ValueError(
                f"global_batch_size {self.global_batch_size} must be divisible by "
                f"sequence_length {self.sequence_length}"
            )
        if self.rank_batch_size % self.sequence_length != 0:
            raise ValueError(
                f"rank_batch_size {self.rank_batch_size} must be divisible by "
                f"sequence_length {self.sequence_length}"
            )
        self.seqs_per_rank = self.rank_batch_size // self.sequence_length
        micro_seqs = max(1, min(self.seqs_per_rank, 8))

        self._regmix_ds = MemmapTokenDataset(
            list(regmix_paths),
            chunk_size=self.sequence_length,
            dtype=regmix_dtype,
        )
        self._math_ds: Optional[MemmapTokenDataset] = None
        if math_paths:
            self._math_ds = MemmapTokenDataset(
                list(math_paths),
                chunk_size=self.sequence_length,
                dtype=math_dtype,
            )

        self._regmix = InfiniteBatchStream(
            self._regmix_ds,
            micro_seqs,
            self.num_workers,
            self.seed,
            self.dp_rank,
            self.dp_world_size,
        )
        self._math: Optional[InfiniteBatchStream] = None
        if self._math_ds is not None:
            self._math = InfiniteBatchStream(
                self._math_ds,
                micro_seqs,
                self.num_workers,
                self.seed + 17,
                self.dp_rank,
                self.dp_world_size,
            )

        end = int(self.phases[-1].end_step) if self.phases else 0
        self._total_steps = int(total_steps) if total_steps is not None else end
        if self._total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self._step_cursor = 0
        self._last_phase: Optional[str] = None
        self._last_math_frac: Optional[float] = None
        self._last_corpus: Optional[str] = None

    @property
    def total_batches(self) -> Optional[int]:
        return self._total_steps

    @property
    def last_phase(self) -> Optional[str]:
        return self._last_phase

    @property
    def last_math_frac(self) -> Optional[float]:
        return self._last_math_frac

    @property
    def last_corpus(self) -> Optional[str]:
        return self._last_corpus

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "epoch": self._epoch,
            "step_cursor": self._step_cursor,
            "regmix": self._regmix.state_dict(),
            "math": self._math.state_dict() if self._math is not None else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.batches_processed = int(state_dict.get("batches_processed", 0))
        self.tokens_processed = int(state_dict.get("tokens_processed", 0))
        self._epoch = state_dict.get("epoch")
        self._step_cursor = int(state_dict.get("step_cursor", self.batches_processed))
        if "regmix" in state_dict and state_dict["regmix"] is not None:
            self._regmix.load_state_dict(state_dict["regmix"])
        if self._math is not None and state_dict.get("math"):
            self._math.load_state_dict(state_dict["math"])

    def reshuffle(self, epoch: Optional[int] = None, **kwargs) -> None:
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else int(self._epoch) + 1
        self._epoch = int(epoch)

    def get_mock_batch(self) -> Dict[str, Any]:
        n = self.seqs_per_rank
        ids = torch.randint(0, 1000, (n, self.sequence_length), dtype=torch.long)
        return {"input_ids": ids}

    def _math_stream(self, step: int) -> InfiniteBatchStream:
        if self._math is None:
            raise RuntimeError(
                f"step {step} needs math corpus but no math paths were staged"
            )
        return self._math

    def _build_step_batch(self, step: int) -> Dict[str, Any]:
        frac = float(math_fraction_for_step(step, self.phases))
        phase = phase_for_step(step, self.phases)
        n = self.seqs_per_rank
        if frac <= 0.0:
            ids = next_rank_input_ids(self._regmix, n)
            corpus = "regmix"
        elif frac >= 1.0:
            ids = next_rank_input_ids(self._math_stream(step), n)
            corpus = "math"
        else:
            n_math = int(round(n * frac))
            n_math = min(max(n_math, 0), n)
            n_reg = n - n_math
            chunks: List[torch.Tensor] = []
            if n_reg:
                chunks.append(next_rank_input_ids(self._regmix, n_reg))
            if n_math:
                chunks.append(next_rank_input_ids(self._math_stream(step), n_math))
            ids = torch.cat(chunks, dim=0)
            corpus = f"mix:{frac:.3f}"
        self._last_phase = phase.name
        self._last_math_frac = frac
        self._last_corpus = corpus
        return {"input_ids": ids}

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        start = self._step_cursor
        for step in range(start, self._total_steps):
            self._step_cursor = step + 1
            yield self._build_step_batch(step)
