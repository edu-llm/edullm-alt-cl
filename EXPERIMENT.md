# Experiment: HQ front-load (alt-CL)

## Hypothesis

Minimizing loss on HQ data early moves the model toward a useful basin. Continuing on general pretraining data from that starting point should help a general-purpose model, even without a single downstream task. Later re-exposure of HQ inside the general mix is allowed; forgetting during the long regmix phase is expected and not treated as a failure mode.

## Model / budget (shared eduLLM 370M contract)

| Knob | Value |
|---|---|
| Architecture | OLMo2 (`TransformerConfig.olmo2_370M`) |
| Params | 370M |
| Tokenizer | `tokenizer/dolma2-bpe` / `allenai/dolma2-tokenizer` |
| Seq len | 2048 |
| Global batch tokens | 4_194_304 |
| Peak LR | 4e-4 |
| LR schedule | warmup then constant (`alpha_f=1.0`) |
| LR warmup | 24 steps ≈ 100M tokens |
| Total tokens | ~10B ≈ 2384 steps |

These match the org’s existing OLMo2-370M / RegMix-10B training contract so runs are comparable to other 10B ladder arms. This repo does **not** live under `edullm-p1`.

## Data phases

| Phase | Steps (approx) | Tokens (approx) | Corpus |
|---|---|---|---|
| A — LR warmup on pretrain | 0–23 | ~100M | `pretrain/regmix-10b` train |
| B — HQ front-load | 24–47 | ~100M | `pretrain/hq-frontload-100m` train (**not published yet**) |
| C — remaining pretrain | 48–2383 | ~9.8B | `pretrain/regmix-10b` train |

- Re-exposure: HQ documents may appear again inside Phase C; no exclusion filter required.
- Control (when you want one): same hparams, flat shuffle over `pretrain/regmix-10b` for all 2384 steps (no Phase B).
- Evals: deferred.

## Published artifacts

| Dataset ID | Role | Status |
|---|---|---|
| `pretrain/regmix-10b` | Phases A + C | live (`v1`) |
| `tokenizer/dolma2-bpe` | tokenizer dependency | live (`v1`) |
| `pretrain/hq-frontload-100m` | Phase B | **design only** — see `DATASET-DESIGN.md` |

Shape A: HQ is its own pretrain token corpus; the trainer switches source by step. No `curriculum/*` token-order required.

## Repo scope (this repository)

1. Design docs (`EXPERIMENT.md`, `DATASET-DESIGN.md`)
2. Trainer: `train_hq_frontload_370m.py` — stages from `s3://edullm-data`, runs phases A→B→C
3. Phase logic: `alt_cl/phases.py` (+ unit tests)
4. Launch: `launch/launch_arm.sh`
5. Scripts to build/tokenize/publish the 100M HQ corpus (once upstream is chosen)

## Blockers

1. **HQ upstream source** — choose and generate ~100M tokens (see `DATASET-DESIGN.md`)
2. **Slice path + held-out carve** for that corpus (irreversible before publish)
