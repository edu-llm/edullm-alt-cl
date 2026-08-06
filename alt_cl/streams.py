"""Memmap token streams used by the curriculum data loader."""

from __future__ import annotations

import textwrap

from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler


class MemmapTokenDataset(Dataset):
    """Contiguous ``chunk_size`` chunks over one or more uint32 token memmaps."""

    def __init__(
        self,
        paths: List[str],
        chunk_size: int,
        dtype: Any = np.uint32,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self._mmaps: List[np.memmap] = []
        self._cum_chunks: List[int] = []
        total = 0
        for p in paths:
            mm = np.memmap(p, mode="r", dtype=dtype)
            n = (len(mm) - 1) // self.chunk_size
            if n <= 0:
                continue
            self._mmaps.append(mm)
            total += n
            self._cum_chunks.append(total)
        if total <= 0:
            raise ValueError(f"No usable chunks in {len(paths)} paths")
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._total
        prev = 0
        for mm, cum in zip(self._mmaps, self._cum_chunks):
            if idx < cum:
                local = idx - prev
                start = local * self.chunk_size
                arr = np.asarray(
                    mm[start : start + self.chunk_size + 1], dtype=np.int64
                )
                return torch.from_numpy(arr[:-1].copy())
            prev = cum
        raise IndexError(idx)


def collate_input_ids(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"input_ids": torch.stack(batch, dim=0)}


class InfiniteBatchStream:
    """Shuffled microbatches that wrap epochs indefinitely."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.pin_memory = torch.cuda.is_available()
        self._epoch = 0
        self._loader: Optional[DataLoader] = None
        self._it: Optional[Iterator] = None

    @property
    def epoch(self) -> int:
        return self._epoch

    def state_dict(self) -> Dict[str, Any]:
        return {"epoch": self._epoch, "seed": self.seed}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._epoch = int(state.get("epoch", 0))
        self._loader = None
        self._it = None

    def _make_loader(self) -> DataLoader:
        if self.world_size > 1:
            sampler: Any = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed,
                drop_last=True,
            )
            sampler.set_epoch(self._epoch)
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch * 1_000_003)
            sampler = RandomSampler(self.dataset, replacement=False, generator=g)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_input_ids,
        )

    def next_batch(self) -> Dict[str, torch.Tensor]:
        if self._it is None:
            self._loader = self._make_loader()
            self._it = iter(self._loader)
        while True:
            try:
                return next(self._it)
            except StopIteration:
                self._epoch += 1
                self._loader = self._make_loader()
                self._it = iter(self._loader)


def next_rank_input_ids(
    stream: InfiniteBatchStream, n_seqs: int
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    got = 0
    while got < n_seqs:
        x = stream.next_batch()["input_ids"]
        chunks.append(x)
        got += x.size(0)
    return torch.cat(chunks, dim=0)[:n_seqs]
