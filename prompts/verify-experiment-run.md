# Agent prompt: verify alt-cl experiment will run

Copy everything below the line into a new agent session.

---

You are verifying that the **edullm-alt-cl** experiment can actually run end-to-end — both **local/ad-hoc** and **eduLLM platform** submit paths. Repo root: the workspace containing `EXPERIMENT.md`, `configs/`, `scripts/`, `alt_cl/`, `.edullm/`, `guides/`, and `artifacts/`.

**Goal:** Find blockers that would stop pretrain and/or SFT from launching correctly (wrong recipe, crash on start, or platform refusal). Prefer evidence (commands, file reads, pytest, optional `aws s3 ls` / MCP) over guesses. Do **not** start a full 10B train or submit a platform run. Do **not** commit or push unless asked.

## Experiment contract (must match code + docs)

1. **Pretrain arms** (OLMo2-370M, Dolma2, via OLMo-core @ `99e0009ed67679c90da970ec5ba439c9459e3757`):
   - `math-front-anneal`: warmup regmix → exclusive math ~100M → bulk regmix → shared end anneal
   - `math-anneal`: warmup → bulk regmix → **same** anneal
   - Anneal: last ~500M tokens (119 steps), linear math mix **0 → ~40%** (~100M math mass)
   - Exact budgets: treatment **2408** steps / `max_tokens: 10099884032`; control **2384** / `9999220736` (exact steps × GBS `4194304`)
   - Configs: `configs/math_front_anneal_10b.yaml`, `configs/math_anneal_10b.yaml`
   - Corpora from YAML (not the form’s single dataset field): `pretrain/regmix-10b` + `pretrain/math-frontload-100m`
   - Entries:
     - Platform: `.edullm/train_pretrain.py` + `--save-folder "$EDULLM_CHECKPOINT_DIR"` (see `guides/platform-submit.md`)
     - Local: `python -m scripts.train_olmo --config …` / `launch/launch_arm.sh`
2. **SFT** (identical for both arms):
   - Mix: `artifacts/math-sft-60m/train.jsonl.gz` (~60M tokens: MetaMath ~42M + no_robots ~2.6M + UltraChat ~15.4M) — or `s3://` via `--sft-path` on platform
   - Config: `configs/math_sft_60m.yaml` — **228** steps, GBS `262144`, `max_tokens: 59768832`
   - Entries: `scripts.train_sft` / `.edullm/train_sft.py`; chat template + **assistant-only** loss
3. **Checkpoints (platform-safe):** `checkpoint_keep_last: null`, `ephemeral_checkpoint_every_steps: null` on arm + SFT configs (smoke may keep `keep_last` for local)
4. **Readout (later):** GSM8K/MATH vs SFT step — eval ladder may still be missing; note separately from “train will run.”

## Checklist (do in order)

### A. Repo + docs + unit tests
- [ ] Confirm git root. Skim `EXPERIMENT.md`, `README.md`, `guides/platform-submit.md` for contradictions with `alt_cl/phases.py`, configs, and `.edullm/` entries.
- [ ] Confirm docs agree that platform uses `dataset_release: none` and `$EDULLM_CHECKPOINT_DIR` (not `edullm-checkpoints` as the live save path).
- [ ] Run: `python -m pytest tests/ -q`
- [ ] Failures here are P0.

### B. Phase schedule integrity
- [ ] Confirm `alt_cl/phases.py`: both arms share anneal shape; treatment has exclusive front; control does not.
- [ ] Confirm anneal window **119**, `anneal_p_max()` ≈ **0.40**, math mass ≈ **100M** tokens.
- [ ] Confirm YAML `phases:` and `train.max_tokens` / `global_batch_size` are exact multiples and match `phase_boundaries` via:
  - `python -m scripts.validate_experiment --config configs/math_front_anneal_10b.yaml`
  - `python -m scripts.validate_experiment --config configs/math_anneal_10b.yaml`
- [ ] Confirm `max_tokens % global_batch_size == 0` (contract enforces this).

### C. Pretrain data readiness
- [ ] Local math frontload exists: `artifacts/math-frontload-100m/` (train/val `.u32le.bin` shards). Read `sample_summary.json`; expect ~99.8M train / ~226k val Dolma2 tokens.
- [ ] Configs point at `pretrain/regmix-10b` and `pretrain/math-frontload-100m`. Check whether math is **published** under `s3://edullm-data/_catalog/pretrain/` (e.g. AWS MCP / `aws s3 ls`). Regmix should already be live.
- [ ] If math is **only local**: stock arm configs will fail staging (`allow_missing_math` defaults false). Workarounds: publish, or `data.math_paths_file` listing local train shards. **A launch that expects published math but only has local artifacts is a P0 for stock configs.**
- [ ] Dry-run (local default save folder):
  - `python -m scripts.train_olmo --config configs/math_front_anneal_10b.yaml`
  - `python -m scripts.train_olmo --config configs/math_anneal_10b.yaml`
  - Plan must show correct steps, dataset ids, `ephemeral_checkpoint_every_steps: null`, fingerprint fields.
- [ ] Dry-run with platform env:
  - `EDULLM_CHECKPOINT_DIR=s3://sbsandbox-intern-edullm-outputs/teams/scratch/runs/run_demo/checkpoints` `python -m scripts.train_olmo --config configs/math_front_anneal_10b.yaml`
  - Plan `save_folder` / `s3_checkpoints` must be that URI (not a local `alt_cl/data/...` path).
- [ ] If `OLMO_ROOT` is available: `python -m scripts.validate_experiment --config … --olmo-root $OLMO_ROOT` and verify revision pin matches YAML.

### D. SFT data + script readiness
- [ ] `artifacts/math-sft-60m/train.jsonl.gz` + `manifest.json` exist; realized totals ~60M; sources metamath / no_robots / ultrachat.
- [ ] Spot-check JSONL rows: `messages` with `user`/`assistant` roles.
- [ ] Dry-run: `python -m scripts.train_sft --config configs/math_sft_60m.yaml --load-path /tmp/dummy_ckpt` — **228** steps, GBS 262144, `ephemeral` null, correct `sft_path`.
- [ ] Dry-run with `--save-folder s3://…/checkpoints` — plan `save_folder` is that URI.
- [ ] Read `alt_cl/sft_data.py` + `alt_cl/olmo_ext/sft_trainer_build.py`: prompt masking; load `--load-path` with fresh optim unless resume; no `mkdir` on `s3://` save folders.
- [ ] If `olmo_core` importable: import `PackedSFTDataLoader` / `build_sft_trainer` seams (no fit). Else record “image/cluster-only.”

### E. Platform submit surface
- [ ] `.edullm/Dockerfile` exists: bare `ARG BASE_IMAGE` (no default), pins torch/OLMo-core/edullm-data, asserts CUDA torch.
- [ ] `.edullm/train_pretrain.py` / `train_sft.py`: default `--save-folder` from `$EDULLM_CHECKPOINT_DIR`; refuse launch without save-folder; dry-run works:
  - `python .edullm/train_pretrain.py local --config configs/smoke.yaml --save-folder s3://demo/checkpoints --dry-run`
- [ ] `alt_cl/platform_env.py`: `resolve_save_folder` precedence CLI > `EDULLM_CHECKPOINT_DIR` > local `output_dir/checkpoints`; remote save does not use S3 as local scratch root.
- [ ] `guides/platform-submit.md` documents `dataset_release: none`, example `bash -lc` command with `"$EDULLM_CHECKPOINT_DIR"`, and platform registration follow-ups.
- [ ] Note: **repo registration in `edu-llm/platform`** (`repositories.yaml` + ECR + workload) is still required for Submit a run — this is a P0 for *platform submit*, not for local dry-run. Confirm whether `edullm-alt-cl` appears in platform `config/repositories.yaml` if you can read that repo.

### F. Local launch plumbing
- [ ] `launch/launch_arm.sh` targets `scripts.train_olmo`; passes `--save-folder` when `EDULLM_CHECKPOINT_DIR` / `SAVE_FOLDER` set; still pretrain-only (not SFT).
- [ ] Deprecated `train_hq_frontload_370m.py` only redirects.
- [ ] Smoke: `python -m scripts.validate_experiment --config configs/smoke.yaml` + dry-run plan (`allow_missing_math: true`).

### G. Optional smoke (only if GPU + OLMo-core + tiny data available)
- [ ] Do **not** run multi-hour jobs. Cap to smoke; respect idle GPU checks in `train_olmo.pin_cuda_visible_devices`.

## Deliverable

Return a short report:

1. **Verdict:** `READY` | `READY WITH LOCAL WORKAROUNDS` | `BLOCKED`
   - Prefer **READY WITH LOCAL WORKAROUNDS** if code/dry-runs pass but math is unpublished and/or platform registration is missing.
   - Use **BLOCKED** only if stock configs cannot launch without a fix (e.g. broken tests, schedule mismatch, missing SFT mix when claiming SFT-ready).
2. **P0 blockers** (would crash, train the wrong recipe, or refuse a platform submit the human expects to work)
3. **P1 gaps** (eval ladder, unpublished math, unregistered platform repo, no `olmo_core` locally, SFT not on S3, etc.)
4. **Commands you ran** + pass/fail
5. **Exact next actions** for a human (≤5 bullets)

Be concrete: cite file paths and config keys. If something is undocumented, say so. Separate **local train readiness** from **platform Submit readiness** when they differ.
