#!/usr/bin/env python3
"""Publish artifacts/math-frontload-100m/tokens via edullm-data airlock."""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "math-frontload-100m"
TOKENS = SOURCE / "tokens"
SUMMARY = SOURCE / "sample_summary.json"
REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"


def _airlock_root(tokens: Path) -> Path:
    """Dir whose only child is tokens/ — build logs must not be in the payload."""
    root = tokens.parent / "_airlock"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    link = root / "tokens"
    try:
        link.symlink_to(tokens.resolve(), target_is_directory=True)
    except OSError:
        # Windows without symlink privilege: directory junction.
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(link), str(tokens.resolve())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return root


def main() -> None:
    if not (TOKENS / "finemath4plus").is_dir():
        raise SystemExit(f"missing token tree under {TOKENS}")
    summary = {}
    if SUMMARY.is_file():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    notes = (
        f"FineMath-4+ deterministic subsample seed=42069666, "
        f"HF revision={summary.get('hf_revision', REVISION)}, "
        f"dolma2_train={summary.get('dolma2_tokens_train')}, "
        f"dolma2_val={summary.get('dolma2_tokens_val')}."
    )
    airlock = _airlock_root(TOKENS)
    # publish() stages to a fixed landing prefix and does not clear it; wipe leftovers
    # from failed attempts so build.log / index.tsv cannot poison Gate A.
    s3 = Boto3S3.default()
    staging_prefix = "pretrain/math-frontload-100m"
    for obj in list(s3.list("edullm-landing", f"_staging/{staging_prefix}/")):
        s3.delete("edullm-landing", obj["key"])

    result = publish(
        str(airlock),
        dataset_id="pretrain/math-frontload-100m",
        purpose=(
            "~100M-token FineMath-4+ sample for OLMo2-370M math front/back-load, "
            "to test whether early math NLL speeds later math SFT (GSM8K/MATH)"
        ),
        profile="pretrain-tokens/v1",
        tokenizer="tokenizer/dolma2-bpe",
        s3=s3,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        about=(
            "Deterministic FineMath-4+ subsample "
            "(seed=42069666, sha256 rank on url+warc_record_offset), Dolma2-tokenized. "
            "Upstream HF token_count is Llama; published counts are Dolma2."
        ),
        sources=[
            {
                "name": "HuggingFaceTB/finemath finemath-4plus",
                "uri": "https://huggingface.co/datasets/HuggingFaceTB/finemath",
                "license": "ODC-By-1.0",
                "scope": "measured-in-this-dataset",
                "tokens": summary.get("dolma2_tokens_train"),
            }
        ],
        license={"id": "ODC-By-1.0", "basis": "declared"},
        notes=notes,
    )
    print(json.dumps(result if not hasattr(result, "__dict__") else str(result), indent=2, default=str))
    print("Published; wait for validator → s3://edullm-data/pretrain/math-frontload-100m/", file=sys.stderr)


if __name__ == "__main__":
    main()
