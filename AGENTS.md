# AGENTS.md

<!-- edullm:begin -->
<!-- Managed by edu-llm/platform. Edit skills/agents-md-block.md there and re-run
     tools/distribute_agent_layer.py; an edit made here is reverted and, until it is,
     tests/test_agent_layer_is_distributed.py is red. Text outside the markers is
     this repository's own and is never touched. -->

## Running anything on a GPU: use `edullm`, never AWS

This codebase is registered with the eduLLM platform. `edullm` is the only supported way to
reach the cluster from a laptop, and it holds no cloud credential of its own: every AWS
credential lives in a workflow whose trust policy pins it to one file on `main`.

**Do not write a script that calls AWS.** No `boto3`, no `aws` CLI, no `curl` at an AWS
endpoint. For the people here who hold no AWS role that fails, and for the few who do it
succeeds and leaves no run anybody can cite, which is the worse of the two.

```bash
uv tool install --force git+https://github.com/edu-llm/platform
edullm --version
```

Unpinned on purpose, and **re-running that line is the upgrade**. Do not reach for `pip` or
`pipx`, and do not reach for `uv tool upgrade`: what it does depends on how the tool was
installed, so from a release note's pinned line it answers `Nothing to upgrade` and exits 0
however far behind the install is. `uv tool install edullm` is the other near miss and uv
answers `not found in the package registry`, because nothing is published to an index under
that name; the line above installs from git.

If this machine installed before the distribution was renamed, run `uv tool uninstall
edullm-platform` **before** the install line and not after. Both own the same `edullm`
executable and uv deletes it along with the old entry, which leaves a healthy-looking
`uv tool list` and no command.

It needs `gh` logged in and a clone with an `origin` remote, and nothing else — no AWS
profile, no SSO session and no VPN, for anything on this path.

| Verb | What it does |
| --- | --- |
| `edullm check` | Prices a submission from this working tree and lists every refusal. Reaches no network. Writes a first `.edullm/run.yaml` where a registered repository has none |
| `edullm submit` | Makes those same checks and then dispatches the submission workflow |
| `edullm status` | Your recent submissions, or one run described |
| `edullm logs` | The last lines one run printed |
| `edullm cancel` | Stops one admitted run, with a reason that goes on the record |
| `edullm data` | The registered corpora, and which of them a run can actually start |
| `edullm add` | Teaches the platform about a repository. Produces a configuration pull request |
| `edullm ask` | Files an ask for something you need. Produces an issue somebody answers |
| `edullm run` | Ships this working tree to a machine of your own and streams back the output of the command after a bare `--` |
| `edullm shell` | A terminal on that same machine, or a notebook on it with `--notebook` |
| `edullm stop` | Ends the machine those two started, and says what it ran up and where your files are |
| `edullm studio` | Opens the Studio space for one `--project` in your browser. Bare, it lists your spaces; `--stop` ends compute and keeps the disk |
| `edullm console` | Opens the AWS console in your browser, signed in as you |

`edullm <verb> --help` prints what each verb takes; this file covers what the help cannot.

`run`, `shell`, `stop`, `studio` and `console` are the exploration route and not the
submission path. Nothing on it is checked against the registry, priced, approved or written to
a lineage record, so what comes off it is a thing somebody saw rather than a result anybody
can cite. Reach for `check` and `submit` for anything meant to count. **`edullm run` and
`edullm shell` leave a machine of your own running, and `edullm stop` is what ends it** — it
terminates rather than stopping, so the machine's own disk goes with it while the scratch
prefix survives for the next one.

**Start with `edullm check --json`.** It costs a fraction of a second, reaches no network and
lists every refusal at once. **Match on `code`** and act on the `detail` beside it, which
names the field and usually the file; the detail is written for a person and gets reworded, so
do not match on it. Exit 0 stands, 1 is refused on the merits, 2 means the command or the
install is wrong, 3 means the platform could not be asked and is the only one worth retrying,
and 130 is an interrupt and wants nothing done about it.

**Read stdout on its own.** The first check in a repository with no `.edullm/run.yaml` writes
one and says so on stderr, so `edullm check --json 2>&1 | ...` turns that note into a parse
error on the one run where you least want one.

Things the refusals will not tell you until they have cost something.

- **The platform takes a commit, not a working tree.** The image is built from the last
  commit, so anything uncommitted is not part of the run, and it is a push to a branch named
  `edullm/<something>` that builds the image at all.
- **For `--dataset`, absent and `none` are different answers.** Pass the literal word `none`
  where the run reads no corpus, which is what a smoke test, a tokenization or an evaluation
  over existing checkpoints does. Only one of the two is a statement.
- **Pick the corpus with `edullm data` and never off a refusal.** It reaches no network and
  carries what a chooser needs per corpus: train tokens, the tokenizer, the shard dtype, the
  licence, and whether a run naming it will start. **That last one is not the same as
  registered, and the gap costs a machine** — a corpus this platform can resolve and the
  training image cannot build is refused by nothing before the money is spent. The
  `unregistered_dataset` refusal prints names and cannot tell you that, and neither can a
  table in a document. `edullm data --json` puts it under `verdict` for a script to branch on.
- **Write the dtype into the command.** The guard behind `bfloat16_not_in_the_hardware` reads
  the text of the command and cannot see a precision the program sets in code, so a card with
  no bfloat16 in hardware refuses the first kernel that needs it — after the run has been
  priced, released, admitted and given a machine. Naming it turns a dead machine into a free
  refusal: `bash -lc 'python train.py train_module.dp_config.param_dtype=bfloat16'`.
- **`edullm status --json` is free and may be polled.** It answers from GitHub and dispatches
  nothing, ever. Read `needs_a_dispatch`: where it is true the rest of the answer has moved
  into AWS, and `edullm status <run-id>` without `--json`, or `edullm logs`, is what pays a
  workflow for it. Those two are slow by construction and neither belongs in a loop.
- **A run id neither form can find is refused rather than asked after.** `run_id_not_found`
  means the window this searched carries no such run, which is not the same as the run not
  existing — a real one can sit outside it. `--ask-aws` on `status` or `logs` buys the certain
  answer and spends a runner for it. `edullm cancel` is the exception and asks AWS every time
  with no flag, because refusing to stop a job that turns out to be running is worse.

**Never quote a price, a runtime bound, a cost ceiling or who has to approve something from
memory or from a document, this one included.** Those live in reviewed configuration that
changes without anybody being told. Run `edullm check --json` and read `cost` and
`approval_class` out of the output.

One skill carries what this cannot: **registering-a-repository**, for when `check` refuses
with `unregistered_repository`. It is not committed here, because this repository is
registered and so is one of the places that refusal cannot arise; it installs once per person,
and [edu-llm/platform's `skills/README.md`](https://github.com/edu-llm/platform/blob/main/skills/README.md)
says where each host reads one from. Everything else about submitting is above, or is in the
`detail` of the refusal you are looking at. There is no skill for it and that is deliberate:
a table of refusal codes here would be a copy of what `edullm check --json` already prints
beside every one of them.

Also never: pass `--force` to get past a refusal, edit `.edullm/run.yaml` to silence a
refusal without reading what it says, or commit a secret into this repository — the image is
built from the commit.
<!-- edullm:end -->
