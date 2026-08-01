# Dataset design: pretrain/hq-frontload-100m

purpose:  ~100M-token HQ corpus for the OLMo2-370M / RegMix-10B front-load phase, to test whether early HQ loss minimization improves final general pretraining
family:   pretrain
profile:  pretrain-tokens/v1   [verified in registry: yes]
name:     hq-frontload-100m    [validate_dataset_id: PASS]

## Irreversible decisions
slice path:       TODO — decide before writing bytes. Prefer `tokens/<source>/<domain>/` if you may later train/measure on a subset; flat `tokens/` only if you are sure you will never want that slice.
heldout source:   TODO — carve val from a different document pool *before* tokenizing (not a post-shuffle split of the same pool). Match regmix-10b’s ~0.15% per-source val carve if sources are multi-domain.
eval status:      n/a

## Layout
tokens/[<source>/<domain>/]<split>-<NNNNN>.u32le.bin   dtype: uint32   ext: .u32le.bin
target shard size: ~500 MB–1 GB            expected shards: few (100M tokens ≈ 400 MB raw uint32)
target train tokens: ~100_000_000 (± a few %)
target val tokens:   TODO (recommend ~0.15% of train if following regmix-10b)

## Dependencies
tokenizer: tokenizer/dolma2-bpe  — MUST be published before this corpus (already live as v1)
parent:    n/a — standalone pretrain corpus (shape A); not a curriculum token-order

## Phase consumers
- Phase B of the alt-CL trainer in this repo reads `pretrain/hq-frontload-100m` train split only.
- Phase A/C read `pretrain/regmix-10b` (already published). Re-exposure of HQ documents inside later regmix is allowed; no dedup against regmix required at publish time.

## Upstream HQ source
TODO — not chosen yet. Blocks generation. Candidate families already in the org (for inspiration only, not a decision):
- RefHQ-style filtered domain pulls (`pretrain/refhq-regmix-5p5b` is 5.5B, wrong scale)
- FineWeb-Edu / Dolmino / other HQ webs
Whatever is chosen must be tokenized with `tokenizer/dolma2-bpe` (same as regmix-10b).

## Deferred (backfillable, don't block)
about / sources[] / license / notes / limitations[]

## publish() call (fill TODOs, then run)

```python
from edullm_data.publish import publish
from edullm_data.s3 import Boto3S3
import datetime

publish(
    "<local dir or s3://edullm-landing/...>",  # staged tokens/ tree
    dataset_id="pretrain/hq-frontload-100m",
    purpose=(
        "~100M-token HQ corpus for the OLMo2-370M / RegMix-10B front-load phase, "
        "to test whether early HQ loss minimization improves final general pretraining"
    ),
    profile="pretrain-tokens/v1",
    tokenizer="tokenizer/dolma2-bpe",
    s3=Boto3S3.default(),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    # group_meta not required for pretrain-tokens/v1 beyond tokenizer=
)
```

Write to `s3://edullm-landing` → validator promotes to `s3://edullm-data`. On reject, read `_REJECTED.json` next to the upload.
