# Experiment: early math basin → easier math SFT

## Hypothesis

Early **exclusive** training on math-purpose data (after normal LR warmup) moves
the model toward a basin that makes **later math SFT** cheaper/faster. A long
general RegMix phase may erase surface math NLL; that is fine. The claim is
tested at **finetune time**, not by end-of-pretrain general scores alone.

Both arms get the **same end-of-pretrain math anneal** (matched midtraining /
recency). Only the treatment also gets early exclusive math.

## Arms (pretrain)

| Arm | Config | Pretrain schedule |
|---|---|---|
| **math-front-anneal** (treatment) | [`configs/math_front_anneal_10b.yaml`](configs/math_front_anneal_10b.yaml) | warmup regmix → **exclusive math ~100M** → bulk regmix → **math anneal** |
| **math-anneal** (control) | [`configs/math_anneal_10b.yaml`](configs/math_anneal_10b.yaml) | warmup regmix → bulk regmix → **same math anneal** |

LR: warmup on the first ~100M regmix, then constant peak (through front math and anneal).

Treatment is +~100M tokens vs control (additive front block). Anneal shape is identical.

| Entry | Command |
|---|---|
| **Platform (preferred)** | `.edullm/train_pretrain.py` + `$EDULLM_CHECKPOINT_DIR` — see [`guides/platform-submit.md`](guides/platform-submit.md) |
| Local / ad-hoc | `python -m scripts.train_olmo --config <yaml> [--olmo-root …] --launch` or [`launch/launch_arm.sh`](launch/launch_arm.sh) |

Schedule code: [`alt_cl/phases.py`](alt_cl/phases.py); mix: `CurriculumDataLoader`. Both corpora come from YAML ids (`pretrain/regmix-10b` + `pretrain/math-frontload-100m`) via `edullm_data` — platform form uses **`dataset_release: none`**.

### End anneal (locked)

Hard 100M math blocks at the end are a cliff. Instead, weave math into the
**last ~500M tokens** with rising concentration:

| Knob | Value | Why |
|---|---|---|
| Window | last **~500M** tokens (119 steps) | Dense late anneal without a pure-math wall |
| Schedule | **linear** math fraction **0 → ~40%** | Mean **~20%** → **~100M math mass** |
| Mixing | per-step sequence mix: `n_math ≈ round(frac × seqs)` from math stream, rest regmix | Stable, deterministic given seed+step |
| Math mass | **~100M** (same for both arms) | Matched late exposure |

Alternatives (same mass, different feel) — only change if you reopen this:

- Longer: last **~1B**, linear **0 → ~20%** (mean 10%)
- Gentler: last **~2B**, linear **0 → ~10%**
- More “late spike”: quadratic ease-in to the same `p_max` (same integral if you solve for peak)

**Keep the front block exclusive** (100% math for ~100M). The hypothesis is about
early *exclusive* basin-finding; softening the front would muddy that.

## Stage 2 — math SFT (identical for every arm)

Same SFT recipe, same steps/LR, starting from each arm’s final pretrain checkpoint.

**Mix (canonical):** `s3://edullm-data/sft/math-sft-60m/v1/`  
**Local build:** `artifacts/math-sft-60m/train.jsonl.gz`

| Source | Tokens (Dolma2) | Examples |
|---|---:|---:|
| MetaMathQA | ~42.0M | 185,867 |
| no_robots | ~2.6M | 9,125 |
| UltraChat 200k | ~15.4M | 12,862 |
| **Total** | **~60.0M** | **207,854** |

- Format: gzipped JSONL rows `{"id","source","messages":[{"role","content"},...]}`
- Rebuild: `python scripts/build_math_sft_60m.py --out-dir artifacts/math-sft-60m`
- Details: `artifacts/math-sft-60m/README.md` + `manifest.json`
- Not on edullm-data — for platform SFT, stage an `s3://` copy the workload role can read (`--sft-path`)

**SFT train:** `scripts/train_sft.py` / `.edullm/train_sft.py` + `configs/math_sft_60m.yaml`

| Knob | Value |
|---|---|
| Mix | `s3://edullm-data/sft/math-sft-60m/v1/` (or local `artifacts/math-sft-60m/`) |
| Template | Dolma2 chat (`<\|im_start\|>…`) |
| Loss | assistant completions only (`labels=-100` on prompts) |
| max_tokens | 59_768_832 (exact 228 × GBS; ~60M budget) |
| GBS | 262_144 (228 steps) |
| LR | 1e-4, warmup 50, cosine `alpha_f=0.1` |
| Checkpoints | every 25 steps; `ephemeral_checkpoint_every_steps: null` (platform-safe) |

```bash
# Local
python -m scripts.train_sft --config configs/math_sft_60m.yaml \
  --load-path <pretrain_arm_final_ckpt> --launch

# Platform — see guides/platform-submit.md
```

Same recipe for both arms; only `--load-path` / save folder change.

## Eval suite (decent, automatic)

Run on the **base** (end of pretrain) and on a **finetune ladder** (every SFT permanent checkpoint + final):

| Benchmark | Role |
|---|---|
| **GSM8K** | Primary — grade-school; 370M-sensitive |
| **Minerva MATH** (`minerva_math` suite) | Secondary — harder |
| Optional: `gsm_symbolic` (`--include-optional`) | Robustness |

**Primary readout for the hypothesis:** GSM8K (and MATH) **vs SFT steps** — treatment should pull ahead earlier / need fewer SFT tokens to hit a target.

**Harness (wired):** org [`edu-llm/olmo-eval-full`](https://github.com/edu-llm/olmo-eval-full) @ pin in [`configs/eval_math_ladder.yaml`](configs/eval_math_ladder.yaml), **`olmo_core` provider** (raw Trainer checkpoints — no HF convert). Planner: [`scripts/eval_ladder.py`](scripts/eval_ladder.py) / [`.edullm/eval_ladder.py`](.edullm/eval_ladder.py); harness YAML: [`configs/eval_harness_olmo_core.yaml`](configs/eval_harness_olmo_core.yaml).

```bash
# Dry-run plan (no GPU / olmo-eval install needed):
python -m scripts.eval_ladder --config configs/eval_math_ladder.yaml \
  --arm math-front-anneal \
  --checkpoint-dir "$SFT_CKPT_DIR" \
  --base-checkpoint "$PRETRAIN_FINAL/step2408"

# Launch on a host/image with olmo-eval + GPU:
python -m scripts.eval_ladder … --launch
```

SFT ladder steps default to the permanent checkpointer grid for 228 / every 25 → `25…200,228` (plus `--base-checkpoint`).

## Math corpus (~100M, re-used) — locked

| Field | Value |
|---|---|
| Source | **FineMath-4+** — `HuggingFaceTB/finemath`, config `finemath-4plus` |
| HF revision | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| Budget | ~100M **Dolma2** tokens (HF `token_count` is Llama — ignore for budget) |
| Local build | `artifacts/math-frontload-100m/` — **99,793,454** train / **226,553** val Dolma2 tokens |
| Sample | Deterministic: `sha256(f"{seed}:{url}:{warc_record_offset}")`, `seed=42069666` |
| Script | `scripts/sample_finemath_frontload.py` |
| Publish as | `pretrain/math-frontload-100m` via `scripts/publish_math_frontload.py` |
| Usage | Front block (treatment) + anneal stream (both); **re-exposure OK** for v1 |
| Bonus | Upstream already 13-gram decontam vs GSM8K / MATH |

Optional later: publish ~200M with disjoint `front/` vs `anneal/` path slices if you want no replay.

## Full pipeline

```text
pretrain arm  →  final ckpt ($EDULLM_CHECKPOINT_DIR)
                    ↓
              identical math SFT
                    ↓
         GSM8K / MATH ladder curves
```

Compare curves: **math-front-anneal** vs **math-anneal**.

## Model / budget (pretrain)

| Knob | Value |
|---|---|
| Architecture | OLMo2 370M (`TransformerConfig.olmo2_370M`) |
| Tokenizer | Dolma2 (`allenai/dolma2-tokenizer` / `tokenizer/dolma2-bpe`) |
| GBS | 4_194_304 |
| LR | 4e-4; warmup 24 steps on regmix; then constant (`alpha_f=1.0`) |
| RegMix schedule | `pretrain/regmix-10b` ≈ 2384 steps |
| Front math | ≈ 24 steps (~100M), exclusive, **treatment only** |
| Anneal | last 119 steps; math frac 0→≈40%; ≈100M math mass, **both arms** |
| Total steps | treatment **2408** / control **2384** |
| max_tokens | treatment **10_099_884_032** / control **9_999_220_736** (exact steps × GBS) |
| Checkpoints | every 125 steps; `checkpoint_keep_last: null`; **no ephemeral** (platform IAM) |
| Save folder (platform) | `$EDULLM_CHECKPOINT_DIR` → `s3://sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/checkpoints/` |
| OLMo-core pin | `edu-llm/OLMo-core` @ `99e0009ed67679c90da970ec5ba439c9459e3757` |

## Artifacts

| ID / path | Role | Status |
|---|---|---|
| `pretrain/regmix-10b` | general pretrain | live on edullm-data |
| `tokenizer/dolma2-bpe` | tokenizer | live |
| `artifacts/math-frontload-100m/` | front + anneal math stream (local) | **built**; publish → `pretrain/math-frontload-100m` |
| `s3://edullm-data/sft/math-sft-60m/v1/` | Stage-2 SFT mix | **published** (Gate A validated) |
| `.edullm/` | platform image + train entries | **done**; needs platform registration |
| `scripts/train_olmo.py` / `train_sft.py` | train entries | **done** |
| Eval ladder | GSM8K/MATH vs SFT step | **wired** — `configs/eval_math_ladder.yaml` + `scripts/eval_ladder.py` (needs `olmo-eval` to launch) |

## Implementation order

1. ~~Freeze SFT mix~~ — `s3://edullm-data/sft/math-sft-60m/v1/` published
2. ~~Build math corpus~~ — local `artifacts/math-frontload-100m/` ready; **publish** to edullm-data
3. ~~Trainer arms with anneal sequence mix~~ — OLMo-core path (`scripts/train_olmo.py` + `configs/`)
4. ~~SFT train entry~~ — `scripts/train_sft.py` + `configs/math_sft_60m.yaml`
5. ~~Platform-shaped submit path~~ — `.edullm/`, `$EDULLM_CHECKPOINT_DIR`, S3-safe trainers
6. Register repo + workload in `edu-llm/platform`; finish math publish
7. ~~Freeze eval harness + GSM8K/MATH ladder~~ — `olmo-eval-full` pin + `scripts/eval_ladder.py`
8. Smoke → full pretrain × 2 → SFT × 2 → run ladder → plot GSM8K vs SFT step

## Blockers

1. Publish `pretrain/math-frontload-100m` (`scripts/publish_math_frontload.py` → landing → validator) — in flight
2. Register `edullm-alt-cl` in [edu-llm/platform](https://github.com/edu-llm/platform) (`repositories.yaml` + ECR + workload profile) — see [`guides/platform-submit.md`](guides/platform-submit.md)
3. ~~Freeze eval harness + ladder command~~ — done; still need GPU/`olmo-eval` to execute and plot curves
