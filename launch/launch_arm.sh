#!/usr/bin/env bash
# Local / ad-hoc launch for one alt-CL pretrain arm.
# For eduLLM platform submits, use .edullm/train_pretrain.py instead (see guides/platform-submit.md).
#
# Required env: OLMO_ROOT when LAUNCH=1 (unless olmo_core is already on PYTHONPATH)
# Optional: CONFIG, SAVE_FOLDER / EDULLM_CHECKPOINT_DIR / OUTPUT_DIR,
#           NPROC, EXTRA_ARGS, RESUME=1, DATA_CACHE_DIR, LAUNCH=0 (dry-run)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/math_front_anneal_10b.yaml}"
NPROC="${NPROC:-1}"
OLMO_ROOT="${OLMO_ROOT:-}"

ARGS=(
  -m scripts.train_olmo
  --config "$CONFIG"
)

if [[ -n "$OLMO_ROOT" ]]; then
  ARGS+=(--olmo-root "$OLMO_ROOT")
fi

# Prefer explicit --save-folder so the flag is visible (matches platform checkpoint guard).
SAVE="${EDULLM_CHECKPOINT_DIR:-${SAVE_FOLDER:-}}"
if [[ -n "$SAVE" ]]; then
  ARGS+=(--save-folder "$SAVE")
  export SAVE_FOLDER="$SAVE"
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  export OUTPUT_DIR
fi
if [[ -n "${EDULLM_RUN_ID:-}" ]]; then
  export EDULLM_RUN_ID
fi

# Local smoke: set LAUNCH=0 to only print the plan JSON
if [[ "${LAUNCH:-1}" == "1" ]]; then
  if [[ -z "$OLMO_ROOT" ]]; then
    if ! python -c "import olmo_core" >/dev/null 2>&1; then
      echo "OLMO_ROOT must point at the pinned edu-llm/OLMo-core checkout (or install olmo_core)" >&2
      exit 1
    fi
  fi
  ARGS+=(--launch)
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

if [[ -n "${DATA_CACHE_DIR:-}" ]]; then
  ARGS+=(--data-cache-dir "$DATA_CACHE_DIR")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

cd "$ROOT"
if [[ "$NPROC" -gt 1 ]]; then
  exec python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "${ARGS[@]}"
else
  if [[ "${LAUNCH:-1}" == "1" ]]; then
    exec python -m torch.distributed.run --standalone --nproc_per_node=1 "${ARGS[@]}"
  else
    exec python "${ARGS[@]}"
  fi
fi
