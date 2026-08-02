"""OLMo-core data loader for packed SFT batches with prompt-masked labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

try:  # pragma: no cover
    from olmo_core.data.data_loader import TextDataLoaderBase

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    TextDataLoaderBase = object  # type: ignore
    _HAS_OLMO = False


def has_olmo_core() -> bool:
    return bool(_HAS_OLMO)


class PackedSFTDataLoader(TextDataLoaderBase):  # type: ignore[misc]
    """Yields ``{input_ids, labels}`` packs for OLMo-core (labels use -100 mask)."""

    def __init__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        sequence_length: int,
        global_batch_size: int,
        work_dir: str | Path,
        seed: int,
        collator: Any,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> None:
        if not _HAS_OLMO:
            raise ImportError("olmo_core is required for PackedSFTDataLoader")
        super().__init__(
            collator=collator,
            work_dir=work_dir,
            global_batch_size=int(global_batch_size),
            dp_world_size=int(dp_world_size),
            dp_rank=int(dp_rank),
            fs_local_rank=fs_local_rank,
        )
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("input_ids/labels must be 2-D and same shape")
        if int(input_ids.shape[1]) != int(sequence_length):
            raise ValueError(
                f"pack width {input_ids.shape[1]} != sequence_length {sequence_length}"
            )
        if self.global_batch_size % int(sequence_length) != 0:
            raise ValueError("global_batch_size must be divisible by sequence_length")
        self.sequence_length = int(sequence_length)
        self.seed = int(seed)
        self.seqs_per_rank = self.rank_batch_size // self.sequence_length
        self._input_ids = input_ids.long().contiguous()
        self._labels = labels.long().contiguous()
        self._n_packs = int(self._input_ids.shape[0])
        if self._n_packs <= 0:
            raise ValueError("no packed SFT sequences")
        max_steps_from_data = self._n_packs // max(1, self.dp_world_size * self.seqs_per_rank)
        self._total_steps = (
            int(total_steps)
            if total_steps is not None
            else max(1, max_steps_from_data)
        )
        self._step_cursor = 0
        self._perm: Optional[torch.Tensor] = None
        self._cursor = 0

    @property
    def total_batches(self) -> Optional[int]:
        return self._total_steps

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "epoch": self._epoch,
            "step_cursor": self._step_cursor,
            "cursor": self._cursor,
            "perm": None if self._perm is None else self._perm.tolist(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.batches_processed = int(state_dict.get("batches_processed", 0))
        self.tokens_processed = int(state_dict.get("tokens_processed", 0))
        self._epoch = state_dict.get("epoch")
        self._step_cursor = int(state_dict.get("step_cursor", self.batches_processed))
        self._cursor = int(state_dict.get("cursor", 0))
        perm = state_dict.get("perm")
        self._perm = None if perm is None else torch.tensor(perm, dtype=torch.long)

    def reshuffle(self, epoch: Optional[int] = None, **kwargs) -> None:
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else int(self._epoch) + 1
        self._epoch = int(epoch)
        g = torch.Generator()
        g.manual_seed(self.seed + int(self._epoch) * 1_000_003 + self.dp_rank)
        self._perm = torch.randperm(self._n_packs, generator=g)
        self._cursor = 0

    def get_mock_batch(self) -> Dict[str, Any]:
        n = self.seqs_per_rank
        ids = torch.randint(0, 1000, (n, self.sequence_length), dtype=torch.long)
        labels = ids.clone()
        labels[:, :8] = -100
        return {"input_ids": ids, "labels": labels}

    def _next_indices(self, n: int) -> torch.Tensor:
        if self._perm is None:
            self.reshuffle(epoch=self._epoch or 1)
        assert self._perm is not None
        out: list[int] = []
        while len(out) < n:
            if self._cursor >= self._n_packs:
                self.reshuffle()
            assert self._perm is not None
            take = min(n - len(out), self._n_packs - self._cursor)
            chunk = self._perm[self._cursor : self._cursor + take].tolist()
            out.extend(int(x) for x in chunk)
            self._cursor += take
        return torch.tensor(out, dtype=torch.long)

    def _build_step_batch(self) -> Dict[str, Any]:
        # Shard packs across DP ranks via offset in the global index stream.
        idx = self._next_indices(self.seqs_per_rank)
        # Shift by rank so ranks see different packs when world>1.
        if self.dp_world_size > 1:
            idx = (idx + self.dp_rank) % self._n_packs
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
        }

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        start = self._step_cursor
        for step in range(start, self._total_steps):
            self._step_cursor = step + 1
            yield self._build_step_batch()
