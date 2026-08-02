# edullm-alt-cl

**Repo:** https://github.com/GMatherne/edullm-alt-cl

OLMo2-370M experiment: early exclusive **math** block (treatment only), then a
shared end-of-pretrain **math anneal**, then **identical math SFT**, measured on
**GSM8K / MATH**.

Pretrain goes through **OLMo-core** (`Trainer` + `CheckpointerCallback` +
`CurriculumDataLoader`), with arm identity in YAML. Platform submits use
[`.edullm/`](.edullm/) and `$EDULLM_CHECKPOINT_DIR` — see
[`guides/platform-submit.md`](guides/platform-submit.md).

| Doc | What |
|---|---|
| [`EXPERIMENT.md`](EXPERIMENT.md) | Hypothesis, arms, SFT/eval plan |
| [`DATASET-DESIGN.md`](DATASET-DESIGN.md) | `pretrain/math-frontload-100m` design + publish |
| [`guides/platform-submit.md`](guides/platform-submit.md) | eduLLM platform Submit a run |
| [`configs/`](configs/) | Per-arm YAML (pinned OLMo-core revision) |

## Pretrain arms

| Arm | Config | Schedule |
|---|---|---|
| `math-front-anneal` | [`configs/math_front_anneal_10b.yaml`](configs/math_front_anneal_10b.yaml) | warmup → **math ~100M exclusive** → bulk regmix → **anneal** |
| `math-anneal` | [`configs/math_anneal_10b.yaml`](configs/math_anneal_10b.yaml) | warmup → bulk regmix → **same anneal** |

Then both: same ~60M math SFT mix → GSM8K/MATH vs SFT step.

## Status

| Piece | State |
|---|---|
| Phase schedule (`alt_cl/phases.py`) | Locked |
| OLMo-core Trainer path | Done — `scripts/train_olmo.py` + `alt_cl/olmo_ext/` |
| Platform entry (`.edullm/`) | Done — needs `edu-llm/platform` registration |
| Math corpus (local) | Built — `artifacts/math-frontload-100m/` (~99.8M Dolma2 train tokens) |
| Math corpus (edullm-data) | Publish in flight — `scripts/publish_math_frontload.py` |
| SFT mix | Published — `s3://edullm-data/sft/math-sft-60m/v1/` (~60M tokens; local build also in `artifacts/math-sft-60m/`) |
| SFT train | Done — `scripts/train_sft.py` + `configs/math_sft_60m.yaml` |
| Eval ladder | **Wired** — `configs/eval_math_ladder.yaml` + `scripts/eval_ladder.py` (olmo-eval-full / `olmo_core`) |

**Docs:** [`EXPERIMENT.md`](EXPERIMENT.md) (contract) · [`guides/platform-submit.md`](guides/platform-submit.md) (Submit a run) · [`DATASET-DESIGN.md`](DATASET-DESIGN.md) (math corpus)

## Submit on the platform

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \
  .edullm/train_pretrain.py "$EDULLM_RUN_ID" \
  --config configs/math_front_anneal_10b.yaml \
  --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

Form: `dataset_release=none`, branch `edullm/**` or `main` with a built image.
Full field list and platform registration follow-ups:
[`guides/platform-submit.md`](guides/platform-submit.md).

## How to run pretrain (local / ad-hoc)

```bash
pip install -r requirements.txt
# Pin OLMo-core to the revision in the YAML:
#   git clone https://github.com/edu-llm/OLMo-core /opt/OLMo-core
#   git -C /opt/OLMo-core checkout 99e0009ed67679c90da970ec5ba439c9459e3757
#   pip install -e /opt/OLMo-core

CFG=configs/math_front_anneal_10b.yaml

python -m scripts.validate_experiment --config "$CFG" --olmo-root /opt/OLMo-core
python -m scripts.train_olmo --config "$CFG"   # dry-run plan JSON

# Full launch needs published pretrain/math-frontload-100m (or local path overrides).
OLMO_ROOT=/opt/OLMo-core CONFIG="$CFG" ./launch/launch_arm.sh
# Checkpoint override (optional; also accepts EDULLM_CHECKPOINT_DIR):
# SAVE_FOLDER=s3://…/checkpoints OLMO_ROOT=… CONFIG="$CFG" ./launch/launch_arm.sh
# Resume:
RESUME=1 OLMO_ROOT=/opt/OLMo-core CONFIG="$CFG" ./launch/launch_arm.sh
```

Smoke / dry-run without math staging: `configs/smoke.yaml` sets `data.allow_missing_math: true`.

## How to run SFT

Identical recipe for both pretrain arms. Point `--load-path` at each arm’s final checkpoint.

```bash
python -m scripts.train_sft \
  --config configs/math_sft_60m.yaml \
  --load-path /path/to/pretrain/final_ckpt   # dry-run plan JSON

OLMO_ROOT=/opt/OLMo-core python -m scripts.train_sft \
  --config configs/math_sft_60m.yaml \
  --load-path /path/to/pretrain/final_ckpt \
  --olmo-root /opt/OLMo-core --launch
```

Uses Dolma2 chat template, assistant-only loss (`labels=-100` on prompts), ~60M tokens @ GBS 262144 (228 steps).

## How to run evals (GSM8K / MATH ladder)

Pin and tasks live in [`configs/eval_math_ladder.yaml`](configs/eval_math_ladder.yaml). Planner shells out to org **`olmo-eval-full`** with the **`olmo_core`** provider (raw Trainer `stepN` dirs).

```bash
# Dry-run plan JSON (no olmo-eval install required):
python -m scripts.eval_ladder --config configs/eval_math_ladder.yaml \
  --arm math-front-anneal \
  --checkpoint-dir /path/to/sft/checkpoints \
  --base-checkpoint /path/to/pretrain/checkpoints/step2408

# Launch (needs olmo-eval on PATH + GPU, or the olmo-eval-full image):
python -m scripts.eval_ladder … --launch

# Optional robustness tasks (gsm_symbolic):
python -m scripts.eval_ladder … --include-optional --launch
```

Primary readout: **GSM8K accuracy vs SFT step**. Default ladder: permanent SFT saves for 228 / every 25 (`25…200,228`) plus `--base-checkpoint`.

`train_hq_frontload_370m.py` is deprecated (prints a redirect).

## Layout

```
.edullm/                 # platform Dockerfile + train_pretrain / train_sft / eval_ladder
configs/                 # pretrain arms + math_sft_60m + eval_math_ladder + smoke
guides/platform-submit.md
alt_cl/
  phases.py              # curriculum schedule
  platform_env.py        # EDULLM_* / save-folder resolution
  eval_ladder.py         # GSM8K/MATH ladder plan + olmo-eval argv
  sft_data.py            # chat template + prompt masking
  streams.py             # memmap token streams
  data.py                # stage from s3://edullm-data
  contract.py            # YAML validate + resume fingerprint
  olmo_ext/              # CurriculumDataLoader, SFT loader/trainer builders
scripts/
  train_olmo.py          # OLMo-core pretrain entry
  train_sft.py           # OLMo-core SFT entry (--load-path + math-sft-60m)
  eval_ladder.py         # olmo-eval-full ladder (dry-run plan / --launch)
  validate_experiment.py
  sample_finemath_frontload.py
  publish_math_frontload.py
  build_math_sft_60m.py
launch/launch_arm.sh     # local ad-hoc (not platform Submit)
artifacts/               # local builds (gitignored)
tests/
```
