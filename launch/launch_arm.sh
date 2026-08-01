#!/usr/bin/env bash
# Launch one alt-CL arm on ephemeral scratch.
# Required env: SAVE_FOLDER, PROGRESS_DIR
# Optional: ARM (hq-frontload|control), ARM_ID, DATA_CACHE_DIR, NPROC, EXTRA_ARGS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:-hq-frontload}"
ARM_ID="${ARM_ID:-$ARM}"
SAVE_FOLDER="${SAVE_FOLDER:?set SAVE_FOLDER}"
PROGRESS_DIR="${PROGRESS_DIR:?set PROGRESS_DIR}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${TMPDIR:-/tmp}/edullm-data-cache-$$}"
NPROC="${NPROC:-1}"

mkdir -p "$SAVE_FOLDER" "$PROGRESS_DIR" "$DATA_CACHE_DIR"

ARGS=(
  "$ROOT/train_hq_frontload_370m.py"
  --arm "$ARM"
  --arm-id "$ARM_ID"
  --save-folder "$SAVE_FOLDER"
  --progress-dir "$PROGRESS_DIR"
  --data-cache-dir "$DATA_CACHE_DIR"
)

# Local smoke: S3_EXPORT=0 ALLOW_LOCAL_ONLY=1
if [[ "${S3_EXPORT:-1}" == "0" || "${ALLOW_LOCAL_ONLY:-0}" == "1" ]]; then
  ARGS+=(--no-s3-export --allow-local-only)
fi

# Optional: skip HQ publish for control or early smoke
if [[ "${ALLOW_MISSING_HQ:-0}" == "1" ]]; then
  ARGS+=(--allow-missing-hq)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

cd "$ROOT"
if [[ "$NPROC" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NPROC" "${ARGS[@]}"
else
  exec python "${ARGS[@]}"
fi
