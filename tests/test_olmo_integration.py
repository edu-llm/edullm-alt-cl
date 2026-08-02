"""Pinned-API smoke checks, skipped when OLMo-core is unavailable locally."""

from __future__ import annotations

import pytest

olmo_core = pytest.importorskip(
    "olmo_core", reason="cluster-only OLMo-core integration"
)


def test_pinned_callback_and_loader_seams_are_present():
    from olmo_core.data.data_loader import TextDataLoaderBase
    from olmo_core.train.callbacks import Callback
    from olmo_core.train.train_module.transformer.train_module import (
        TransformerTrainModule,
    )

    from alt_cl.olmo_ext.callbacks import PhaseMetricsCallback
    from alt_cl.olmo_ext.curriculum_loader import CurriculumDataLoader, has_olmo_core
    from alt_cl.olmo_ext.trainer_build import build_trainer

    assert has_olmo_core()
    assert issubclass(PhaseMetricsCallback, Callback)
    assert issubclass(CurriculumDataLoader, TextDataLoaderBase)
    assert callable(build_trainer)
    assert hasattr(PhaseMetricsCallback, "log_metrics")
    assert hasattr(PhaseMetricsCallback, "post_attach")
    assert hasattr(CurriculumDataLoader, "_iter_batches")
    assert hasattr(CurriculumDataLoader, "state_dict")
    assert TransformerTrainModule is not None
