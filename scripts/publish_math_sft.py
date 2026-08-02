#!/usr/bin/env python3
"""Stage + publish artifacts/math-sft-60m as sft/math-sft-60m via edullm airlock.

Builds conversations/{train,val}-00000.jsonl.gz (val ~1% by deterministic hash,
disjoint from train) then publish() → edullm-landing → validator → edullm-data.

Requires AWS creds that can write edullm-landing (e.g. AWS_PROFILE=sbsandbox).
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MIX = ROOT / "artifacts" / "math-sft-60m" / "train.jsonl.gz"
SOURCE_MANIFEST = ROOT / "artifacts" / "math-sft-60m" / "manifest.json"
STAGE = ROOT / "artifacts" / "math-sft-60m" / "_airlock"
DATASET_ID = "sft/math-sft-60m"
SEED = 42069666
VAL_FRAC = 0.01


def _iter_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
                yield obj


def _dedup_key(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for m in row.get("messages") or []:
        if isinstance(m, dict):
            parts.append(f"{m.get('role', '')}\x1f{m.get('content', '')}")
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def _is_val(key: str, frac: float = VAL_FRAC) -> bool:
    # Stable fraction from hash prefix.
    n = int(key[:8], 16) / 0xFFFFFFFF
    return n < frac


def _write_jsonl_gz(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            out = {
                "id": row.get("id"),
                "source": row.get("source"),
                "messages": row["messages"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def stage_conversations() -> Tuple[int, int, int]:
    if not SOURCE_MIX.is_file():
        raise SystemExit(f"missing mix: {SOURCE_MIX} (run scripts/build_math_sft_60m.py)")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    conv = STAGE / "conversations"
    conv.mkdir(parents=True)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    seen: set[str] = set()
    n_dup = 0
    for row in _iter_rows(SOURCE_MIX):
        key = _dedup_key(row)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if _is_val(key):
            val.append(row)
        else:
            train.append(row)

    if not train or not val:
        raise SystemExit(f"bad split: train={len(train)} val={len(val)}")

    _write_jsonl_gz(conv / "train-00000.jsonl.gz", train)
    _write_jsonl_gz(conv / "val-00000.jsonl.gz", val)
    return len(train), len(val), n_dup


def main() -> None:
    n_train, n_val, n_dup = stage_conversations()
    mix_meta = {}
    if SOURCE_MANIFEST.is_file():
        mix_meta = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))

    s3 = Boto3S3.default()
    staging_prefix = DATASET_ID
    for obj in list(s3.list("edullm-landing", f"_staging/{staging_prefix}/")):
        s3.delete("edullm-landing", obj["key"])

    purpose = (
        "~60M-token math SFT mix (70% MetaMathQA / ~2.6M no_robots / ~15.4M UltraChat) "
        "for OLMo2-370M alt-cl Stage-2 finetune after math-front-anneal pretrain"
    )
    about = (
        "Deterministic mix (seed=42069666) of MetaMathQA, HuggingFaceH4/no_robots, and "
        "UltraChat 200k train_sft. Rows are chat messages[]; Dolma2 chat template + "
        "assistant-only loss applied at train time. Val is a 1% hash holdout for the "
        "edullm sft-conversations profile (not used as the experiment's primary metric)."
    )
    notes = (
        f"staged_train_rows={n_train}, staged_val_rows={n_val}, "
        f"dropped_exact_dups={n_dup}, "
        f"local_mix_tokens={mix_meta.get('realized', {}).get('total_tokens')}, "
        f"hf_revisions={mix_meta.get('revisions')}"
    )

    result = publish(
        str(STAGE),
        dataset_id=DATASET_ID,
        purpose=purpose,
        profile="sft-conversations/v1",
        s3=s3,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        about=about,
        notes=notes,
        sources=[
            {
                "name": "meta-math/MetaMathQA",
                "uri": "https://huggingface.co/datasets/meta-math/MetaMathQA",
                "license": "MIT",
                "scope": "measured-in-this-dataset",
                "tokens": (mix_meta.get("realized") or {}).get("metamath_tokens"),
            },
            {
                "name": "HuggingFaceH4/no_robots",
                "uri": "https://huggingface.co/datasets/HuggingFaceH4/no_robots",
                "license": "CC-BY-NC-4.0",
                "scope": "measured-in-this-dataset",
                "tokens": (mix_meta.get("realized") or {}).get("no_robots_tokens"),
            },
            {
                "name": "HuggingFaceH4/ultrachat_200k",
                "uri": "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k",
                "license": "MIT",
                "scope": "measured-in-this-dataset",
                "tokens": (mix_meta.get("realized") or {}).get("ultrachat_tokens"),
            },
        ],
        license={"id": "mixed", "basis": "declared"},
        group_meta={
            "conversations": {
                "record_schema": {
                    "type": "object",
                    "required": ["messages"],
                },
                "partitions": [
                    {
                        "name": "train",
                        "by": "path",
                        "glob": "conversations/train-*.jsonl.gz",
                    },
                    {
                        "name": "val",
                        "by": "path",
                        "glob": "conversations/val-*.jsonl.gz",
                    },
                ],
                "dedup": {
                    "method": "sha256-messages",
                    "key": "role+content concatenation",
                    "exact_dups_dropped_at_stage": n_dup,
                },
                "leakage": {
                    "train_val_overlap": 0,
                    "method": "sha256-messages",
                    "note": "val carved by hash fraction; disjoint by construction",
                },
            }
        },
    )
    print(
        json.dumps(
            result if not hasattr(result, "__dict__") else str(result),
            indent=2,
            default=str,
        )
    )
    print(
        f"Published plan; wait for validator → s3://edullm-data/{DATASET_ID}/",
        file=sys.stderr,
    )
    print(f"staged rows train={n_train} val={n_val} dups_dropped={n_dup}", file=sys.stderr)


if __name__ == "__main__":
    main()
