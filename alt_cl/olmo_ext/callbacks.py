"""OLMo-core callbacks for alt-cl phase + train-loss logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

log = logging.getLogger("alt_cl.olmo_ext.callbacks")

try:  # pragma: no cover - OLMo-core absent from local CI.
    from olmo_core.distributed.utils import get_rank
    from olmo_core.train.callbacks import Callback

    _HAS_OLMO = True
except Exception:  # pragma: no cover
    Callback = object  # type: ignore
    get_rank = lambda: 0  # type: ignore
    _HAS_OLMO = False


class PhaseMetricsCallback(Callback):  # type: ignore[misc]
    """Persist per-step phase / math_frac / CE for the alt-cl curriculum."""

    # Run before CheckpointerCallback (priority 1) so metrics match the step.
    priority: ClassVar[int] = 2

    def __init__(self, *, metrics_dir: str | Path, log_interval: int = 10) -> None:
        if not _HAS_OLMO:
            raise ImportError("olmo_core is required for PhaseMetricsCallback")
        super().__init__()  # type: ignore[misc]
        self.metrics_dir = Path(metrics_dir)
        self.log_interval = int(log_interval)
        self._path = self.metrics_dir / "train_loss.jsonl"
        self._progress_path = self.metrics_dir.parent / "progress.json"

    def post_attach(self) -> None:  # pragma: no cover
        if get_rank() == 0:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def _phase_fields(self) -> Dict[str, Any]:
        dl = self.trainer.data_loader
        return {
            "phase": getattr(dl, "last_phase", None),
            "math_frac": getattr(dl, "last_math_frac", None),
            "corpus": getattr(dl, "last_corpus", None),
        }

    def _ce_loss(self, metrics: Dict[str, float]) -> Optional[float]:
        for key in (
            "train/CE loss",
            "train/ce_loss",
            "CE loss",
            "ce_loss",
            "train/CrossEntropyLoss",
        ):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        for key, val in metrics.items():
            low = key.lower()
            if "ce" in low and "loss" in low:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:  # pragma: no cover
        if get_rank() != 0:
            return
        if int(step) % int(self.log_interval) != 0 and int(step) != 1:
            return
        phase = self._phase_fields()
        row: Dict[str, Any] = {
            "step": int(step),
            "phase": phase["phase"],
            "corpus": phase["corpus"],
            "math_frac": phase["math_frac"],
            "loss": self._ce_loss(metrics),
        }
        for key in ("optim/total lr [DP rank 0]", "optim/lr", "lr"):
            if key in metrics:
                try:
                    row["lr"] = float(metrics[key])
                except (TypeError, ValueError):
                    pass
                break
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        try:
            max_steps = getattr(self.trainer, "max_steps", None)
            pct = None
            if max_steps:
                pct = round(100.0 * int(step) / int(max_steps), 4)
            self._progress_path.write_text(
                json.dumps({**row, "total_steps": max_steps, "pct": pct}) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to write progress.json: %s", exc)

    def state_dict(self) -> Dict[str, Any]:
        return {"version": 1}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict
