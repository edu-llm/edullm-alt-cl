# Submit on the eduLLM platform

How to run **edullm-alt-cl** (math front-load + anneal → SFT) on
[edu-llm/platform](https://github.com/edu-llm/platform).

This repo already has a platform image ([`.edullm/Dockerfile`](../.edullm/Dockerfile))
and entries ([`train_pretrain.py`](../.edullm/train_pretrain.py),
[`train_sft.py`](../.edullm/train_sft.py)).
**Registration in `edu-llm/platform` is still required** before Submit a run can
name this repository.

Experiment contract: [`EXPERIMENT.md`](../EXPERIMENT.md). Local ad-hoc launch:
[`README.md`](../README.md) + [`launch/launch_arm.sh`](../launch/launch_arm.sh).

## What the container gets

| Variable | Meaning |
|---|---|
| `EDULLM_RUN_ID` | Stable run id (Batch retries reuse it) |
| `EDULLM_CHECKPOINT_DIR` | `s3://sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/checkpoints/` |
| `EDULLM_OUTPUT_PREFIX` | `s3://sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/` |

Checkpoints **must** go to `"$EDULLM_CHECKPOINT_DIR"` on the command line. The
platform checkpoint guard reads the command text; putting the path only inside
Python is refused. Local work/metrics stay under `/tmp/edullm-alt-cl/…` (not on
that S3 prefix).

Configs set `ephemeral_checkpoint_every_steps: null` and
`checkpoint_keep_last: null` so OLMo-core does not prune objects the workload
role cannot delete.

## Why `dataset_release: none`

Pretrain needs **two** corpora (`pretrain/regmix-10b` +
`pretrain/math-frontload-100m`). The form injects only one `EDULLM_DATASET_*`
triple. Both IDs are taken from the YAML and resolved with `edullm_data` at
runtime — pick **`none`** on the form so the record does not invent a single
corpus.

## Form fields (pretrain)

| Field | Value |
|---|---|
| `repository` | `edullm-alt-cl` (after platform registration) |
| `commit_sha` | full SHA on an `edullm/**` or `main` branch with a built image |
| `dataset_release` | **`none`** |
| `team` | e.g. `pre-training` or `scratch` |
| `experiment` | free text, e.g. `math-front-anneal-v1` |
| `wandb_project` | your W&B project on the eduLLM team |
| `command` | see below (must include `bash -lc` and `"$EDULLM_CHECKPOINT_DIR"`) |

Workload / compute: routine training profiles are 12h / team-lead approval.
`gpu-8xa100` is an **admin exception** (>$20/hr). There is no dedicated
8-GPU training workload profile yet — you may need an advanced
`compute_profile` override plus matching `--nproc-per-node`.

### Example command (8×A100)

Treatment:

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone .edullm/train_pretrain.py "$EDULLM_RUN_ID" --config configs/math_front_anneal_10b.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

Control: same line with `configs/math_anneal_10b.yaml`.

Dry-run (plan JSON only, no fit):

```bash
bash -lc 'python .edullm/train_pretrain.py "$EDULLM_RUN_ID" --config configs/math_front_anneal_10b.yaml --save-folder "$EDULLM_CHECKPOINT_DIR" --dry-run'
```

### SFT

1. Stage `artifacts/math-sft-60m/train.jsonl.gz` to an `s3://` URI the workload
   role can read (not published to edullm-data).
2. Point `--load-path` at the pretrain arm’s final checkpoint under that run’s
   `EDULLM_CHECKPOINT_DIR`.

```bash
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone .edullm/train_sft.py "$EDULLM_RUN_ID" --config configs/math_sft_60m.yaml --load-path "$PRETRAIN_CKPT" --sft-path "$SFT_JSONL_GZ" --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

### Eval ladder (GSM8K / Minerva MATH)

After each arm’s SFT finishes, score **base (pretrain final)** + SFT permanent steps with org **`olmo-eval-full`** (`olmo_core` provider). Planner in this repo; execution needs `olmo-eval` on PATH (or submit against the registered `olmo-eval-full` image).

Dry-run (plan JSON only):

```bash
bash -lc 'python .edullm/eval_ladder.py "$EDULLM_RUN_ID" --config configs/eval_math_ladder.yaml --arm math-front-anneal --checkpoint-dir "$SFT_CKPT_DIR" --base-checkpoint "$PRETRAIN_FINAL" --output-dir "$EDULLM_OUTPUT_PREFIX/eval" --dry-run'
```

Launch (same command without `--dry-run`, on a GPU workload that has `olmo-eval`):

```bash
bash -lc 'python .edullm/eval_ladder.py "$EDULLM_RUN_ID" --config configs/eval_math_ladder.yaml --arm math-front-anneal --checkpoint-dir "$SFT_CKPT_DIR" --base-checkpoint "$PRETRAIN_FINAL" --output-dir "$EDULLM_OUTPUT_PREFIX/eval"'
```

Pin / tasks: [`configs/eval_math_ladder.yaml`](../configs/eval_math_ladder.yaml). Primary readout: GSM8K vs SFT step.

## Image build

Push an `edullm/**` (or `main`) branch so the platform build runs
[`.edullm/Dockerfile`](../.edullm/Dockerfile) (torch 2.9.0 CUDA, OLMo-core
`99e0009…`, edullm-data `v0.6.3`, this package). Wait for the ECR vulnerability
scan (~10 minutes) before submitting. A green build is not enough if the scan
is still running.

## Platform-side follow-ups (not in this repo)

1. Register `edullm-alt-cl` in `edu-llm/platform` `config/repositories.yaml` + ECR + image build.
2. Add a workload profile (12h routine ceiling; A100 = exception).
3. Optionally list `math-frontload-100m` under `config/datasets.yaml` for lineage (optional when `dataset_release: none`).
4. Confirm `pretrain/math-frontload-100m` is validated on `s3://edullm-data` before a full launch.

Until (1)–(2) land, submissions naming this repository are denied as
`unregistered_repository`.
