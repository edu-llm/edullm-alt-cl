# edullm-alt-cl

Alternate curriculum for **OLMo2-370M** on **RegMix-10B**: LR warmup on general pretrain → short HQ front-load (~100M tokens) → rest of pretrain.

| Doc | What |
|---|---|
| [`EXPERIMENT.md`](EXPERIMENT.md) | Hypothesis, phase schedule, blockers |
| [`DATASET-DESIGN.md`](DATASET-DESIGN.md) | Unpublished HQ corpus (`pretrain/hq-frontload-100m`) |

## Schedule

| Phase | Steps | Tokens | Corpus |
|---|---|---|---|
| A | 0–23 | ~100M | `pretrain/regmix-10b` |
| B | 24–47 | ~100M | `pretrain/hq-frontload-100m` (**not published yet**) |
| C | 48–2383 | ~9.8B | `pretrain/regmix-10b` |

Control arm: shuffled regmix for all 2384 steps.

## Setup

```bash
pip install -r requirements.txt
pytest
```

## Train

```bash
RUN_DIR="${TMPDIR:-/tmp}/alt-cl-$$"
mkdir -p "$RUN_DIR"/{ckpts,progress,cache}

# Full arm (needs published HQ + AWS creds for edullm-data / checkpoints)
ARM=hq-frontload \
  SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
  DATA_CACHE_DIR=$RUN_DIR/cache \
  bash launch/launch_arm.sh

# Control
ARM=control ARM_ID=control \
  SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
  DATA_CACHE_DIR=$RUN_DIR/cache \
  bash launch/launch_arm.sh

# Local smoke (no S3 durable export; HQ may be missing)
S3_EXPORT=0 ALLOW_LOCAL_ONLY=1 ALLOW_MISSING_HQ=1 \
  SAVE_FOLDER=$RUN_DIR/ckpts PROGRESS_DIR=$RUN_DIR/progress \
  bash launch/launch_arm.sh
```

Or call the trainer directly:

```bash
python train_hq_frontload_370m.py \
  --arm hq-frontload \
  --save-folder /tmp/ckpts --progress-dir /tmp/progress \
  --no-s3-export --allow-local-only
```

Durable layout: `s3://edullm-checkpoints/alt-cl/<arm_id>/`.

## Status

- Phase schedule + trainer + unit tests: in repo
- HQ corpus upstream / publish: still open (see `DATASET-DESIGN.md`)
- Evals: deferred
