"""CPU tests for memmap streams + mix math used by CurriculumDataLoader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alt_cl.phases import math_fraction_for_step, phase_boundaries
from alt_cl.streams import (
    InfiniteBatchStream,
    MemmapTokenDataset,
    next_rank_input_ids,
)


def _write_u32(path: Path, n_tokens: int, *, start: int = 0) -> None:
    arr = np.arange(start, start + n_tokens, dtype=np.uint32)
    path.write_bytes(arr.tobytes())


def test_memmap_dataset_chunks(tmp_path: Path):
    p = tmp_path / "a.u32le.bin"
    # 10 tokens → with chunk_size=4 need chunk+1 lookahead → 2 chunks (0..4, 4..8)
    _write_u32(p, 10)
    ds = MemmapTokenDataset([str(p)], chunk_size=4)
    assert len(ds) == 2
    x0 = ds[0]
    assert x0.shape == (4,)
    assert x0.dtype == torch.int64
    assert x0.tolist() == [0, 1, 2, 3]


def test_infinite_stream_state_roundtrip(tmp_path: Path):
    p = tmp_path / "b.u32le.bin"
    _write_u32(p, 64)
    ds = MemmapTokenDataset([str(p)], chunk_size=4)
    stream = InfiniteBatchStream(ds, batch_size=2, num_workers=0, seed=7)
    _ = stream.next_batch()
    stream._epoch = 3
    state = stream.state_dict()
    stream2 = InfiniteBatchStream(ds, batch_size=2, num_workers=0, seed=7)
    stream2.load_state_dict(state)
    assert stream2.epoch == 3


def test_next_rank_input_ids_shape(tmp_path: Path):
    p = tmp_path / "c.u32le.bin"
    _write_u32(p, 256)
    ds = MemmapTokenDataset([str(p)], chunk_size=8)
    stream = InfiniteBatchStream(ds, batch_size=2, num_workers=0, seed=1)
    ids = next_rank_input_ids(stream, 5)
    assert ids.shape == (5, 8)


def test_mix_rounding_matches_phase_schedule():
    """Document the locked per-step mix: n_math = round(frac * n)."""
    phases = phase_boundaries("math-front-anneal")
    anneal = [p for p in phases if p.name == "D_anneal"][0]
    n = 32
    for step in range(anneal.start_step, anneal.end_step):
        frac = math_fraction_for_step(step, phases)
        n_math = int(round(n * frac))
        assert 0 <= n_math <= n
        if step == anneal.start_step:
            assert n_math == 0
        if step == anneal.end_step - 1:
            assert n_math == int(round(n * anneal.math_frac_end))


def test_curriculum_loader_import_guard_without_olmo():
    """Module must import; constructing loader requires olmo_core."""
    from alt_cl.olmo_ext import curriculum_loader as cl

    if cl.has_olmo_core():
        pytest.skip("olmo_core present; guard path not exercised")
    with pytest.raises(ImportError):
        cl.CurriculumDataLoader(
            regmix_paths=["x"],
            math_paths=None,
            phases=phase_boundaries("math-anneal"),
            sequence_length=4,
            global_batch_size=16,
            work_dir=".",
            seed=0,
            collator=object(),
        )
