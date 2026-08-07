# Proving that this repository's trainer resumes

The platform refuses a submission asking for more than one attempt against a repository nobody
has watched resume. This file is how you clear that for `edullm-alt-cl`, and it lives here rather
than in the platform's documentation because the flags it uses are this trainer's.

The standard is not "the step counter went up". It is that a second process's **loss values match
the first run's at the steps they share**. A restored model with an unrestored data loader
produces a plausible curve that diverges on its first step, and a restarted learning-rate
schedule reports a number that looks like a number. Matching losses over dozens of steps is what
rules both out.

## What this trainer does, which shapes the proof

`.edullm/train_pretrain.py` takes `--save-folder` and `--resume` and **rejects any argument it
does not recognise** (`parse_known_args`, then exit 2). There is no `--load-path` and no way to
pass a config override through it, so the OLMo-core form of this proof -- a second submission
carrying `trainer.load_path=<another run's prefix>` -- cannot be expressed here.

What it does instead is in `scripts/train_olmo.py`:

```python
soft_remote = bool(platform_run_id() and is_remote_uri(plan["save_folder"]))
trainer_resume = True if soft_remote else resume
```

On the platform, with a remote save folder, **resume is always on** and reads whatever is under
`$EDULLM_CHECKPOINT_DIR`. That is written for the case the attempt count actually buys: a Batch
retry reuses the same checkpoint directory. `_prepare_run_dir` then compares
`run_fingerprint.json` against the current config and exits rather than resuming onto a different
one, so a config change cannot be mistaken for a resume.

So the proof has to put a checkpoint under the second process's own `$EDULLM_CHECKPOINT_DIR`.

## Run A: the source run

One submission, `edullm-alt-cl-train`, `gpu-1xa10g`, `--hours 1`, this config. Nothing special.

```
python .edullm/train_pretrain.py --config configs/resume_proof.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"
```

It trains 44 steps and writes a checkpoint every 4, keeping all of them. Record its run id and
its per-step losses.

## Run B: a separate submission, resuming mid-schedule

A second submission with **its own run id and its own empty checkpoint directory**, whose command
seeds that directory from a checkpoint in the middle of run A's schedule and then trains
normally. Seeding rather than pointing is what keeps this expressible: the command still expands
`$EDULLM_CHECKPOINT_DIR`, which the platform's checkpoint guard requires, and run B still writes
only to its own prefix, so the two runs' outputs do not mingle.

```
sh -c 'python -c "
import boto3, os
s3 = boto3.client(\"s3\")
src = os.environ[\"RESUME_PROOF_SOURCE\"]
dst = os.environ[\"EDULLM_CHECKPOINT_DIR\"]
sb, sp = src.removeprefix(\"s3://\").split(\"/\", 1)
db, dp = dst.removeprefix(\"s3://\").split(\"/\", 1)
for page in s3.get_paginator(\"list_objects_v2\").paginate(Bucket=sb, Prefix=sp):
    for o in page.get(\"Contents\", []):
        s3.copy_object(Bucket=db, Key=dp + o[\"Key\"][len(sp):], CopySource={\"Bucket\": sb, \"Key\": o[\"Key\"]})
" && python .edullm/train_pretrain.py --config configs/resume_proof.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

`RESUME_PROOF_SOURCE` is run A's prefix for one mid-schedule checkpoint plus its
`run_fingerprint.json`. Seed a checkpoint from the middle rather than the last one: the number of
steps the two runs share is the strength of the evidence, and seeding the final checkpoint leaves
almost none.

## What to read, and what to write down

From run B's log: the step it reports loading, and its first logged step. From both logs: the loss
at every step they share. The entry in the platform's
`config/reports/resume-demonstrations.yaml` cites run B's id, the commit its image was built
from, the step it loaded and the step it reached — two numbers a reviewer opens the run and reads.

If the losses do **not** match, that is the finding, and it belongs in a bug report rather than in
a demonstration entry. It would mean this repository's two declared attempts cannot help it: a
retry would restart from zero and report success.
