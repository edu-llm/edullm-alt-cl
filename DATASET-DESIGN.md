# Dataset design: pretrain/math-frontload-100m

purpose:  ~100M-token FineMath-4+ sample for OLMo2-370M early exclusive math + shared end anneal, to test whether early math NLL speeds later math SFT (GSM8K/MATH)
family:   pretrain
profile:  pretrain-tokens/v1   [verified in registry: yes]
name:     math-frontload-100m  [validate_dataset_id: PASS — re-verify before publish]

**Local build status:** tokens live under `artifacts/math-frontload-100m/`
(~99.8M train Dolma2 tokens; see `sample_summary.json`). Publish target:
`pretrain/math-frontload-100m` on `s3://edullm-data` via
`scripts/publish_math_frontload.py`. Training resolves that id from YAML (platform
form: `dataset_release: none`) — see [`guides/platform-submit.md`](guides/platform-submit.md).

## Irreversible decisions
slice path:       `tokens/finemath4plus/` — single upstream; nest kept for clarity / future mixes
heldout source:   carve **0.15%** of selected docs (by same deterministic rank, last docs) into `val-*.u32le.bin` before packing train; docs disjoint
eval status:      n/a

## Layout
tokens/finemath4plus/<split>-<NNNNN>.u32le.bin   dtype: uint32   ext: .u32le.bin
target shard size: ~500 MB–1 GB
target train tokens: **100_000_000** Dolma2 tokens (±2%)
target val tokens:   ~0.15% of selected pool

## Dependencies
tokenizer: tokenizer/dolma2-bpe
parent:    n/a

## Upstream (locked)

| Field | Value |
|---|---|
| HF dataset | `HuggingFaceTB/finemath` |
| Config | **`finemath-4plus`** (FineMath-4+) |
| License | ODC-By-1.0 (+ Common Crawl ToU) |
| Note | HF `token_count` is **Llama** tokens — budget accounting uses **Dolma2** counts after tokenize |
| Decontam | Upstream already 13-gram decontam vs GSM8K / MATH / MMLU / ARC |

**Pin HF revision:** `e92b25a616738fe95dc186b64dfb19f9c8525594` (record in `sample_config.json`).

## Reproducible ~100M sample

Script: `scripts/sample_finemath_frontload.py`

1. Load `finemath-4plus` train at pinned revision.
2. Rank every doc by  
   `key = sha256(f"{SEED}:{url}:{warc_record_offset}")`  
   ascending hex digest (`SEED=42069666`).
3. Walk that order; tokenize each `text` with Dolma2; append until **≥ 100_000_000** train tokens (stop at first doc that would exceed by >2%, or accept small overshoot — script flag).
4. Last **0.15%** of *accepted doc list* (by rank order) → val; remainder → train.
5. Pack headerless `.u32le.bin` shards; write `sample_manifest.jsonl` (url, offset, dolma2_tokens, split) for audit.

Same seed + revision + tokenizer ⇒ same shard bytes.

## Deferred (backfillable)
about / sources[] / license / notes / limitations[]

## publish() call

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime

publish(
    "<dir containing tokens/>",
    dataset_id="pretrain/math-frontload-100m",
    purpose=(
        "~100M-token FineMath-4+ sample for OLMo2-370M math front-load + end anneal, "
        "to test whether early math NLL speeds later math SFT (GSM8K/MATH)"
    ),
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/dolma2-bpe",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    about=(
        "Deterministic FineMath-4+ subsample (seed=42069666, sha256 rank on url+warc_record_offset), "
        "Dolma2-tokenized. Upstream HF token_count is Llama; published counts are Dolma2."
    ),
    sources=[{
        "name": "HuggingFaceTB/finemath finemath-4plus",
        "uri": "https://huggingface.co/datasets/HuggingFaceTB/finemath",
        "license": "ODC-By-1.0",
        "scope": "measured-in-this-dataset",
    }],
    license={"id": "ODC-By-1.0", "basis": "declared"},
    notes="Pin HF revision in sample_manifest / notes at build time.",
)
```
