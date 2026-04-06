#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_root> <code>" >&2
  exit 2
fi

RUN_ROOT="$1"
CODE="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MANIFEST="${RUN_ROOT}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

ensure_run_dirs "${RUN_ROOT}"

TRAIN_SCRIPT="${RELDEC_DIR}/train_reldec.py"
CHECKPOINT_DIR="${RUN_ROOT}/${CODE}/checkpoints"
LOG_FILE="${RUN_ROOT}/logs/train_${CODE}.log"
STATE_FILE="${RUN_ROOT}/state/train_${CODE}.json"
PID_FILE="${RUN_ROOT}/pids/train_${CODE}.pid"

mkdir -p "${CHECKPOINT_DIR}" "${RUN_ROOT}/${CODE}/results"

EPISODES_PER_SNR="$(read_manifest_field "${MANIFEST}" "train.episodes_per_snr")"
CHECKPOINT_EVERY="$(read_manifest_field "${MANIFEST}" "train.checkpoint_every_episodes")"
LOG_EVERY="$(read_manifest_field "${MANIFEST}" "train.log_every")"
MAX_EPISODES="$(read_manifest_field "${MANIFEST}" "train.max_episodes")"
SEED="$(read_manifest_field "${MANIFEST}" "train.seeds.${CODE}")"

LATEST_CKPT="${CHECKPOINT_DIR}/checkpoint_latest.npz"

EXTRA_START='{"component":"train"}'
write_state_json "${STATE_FILE}" "starting" "${CODE}" "$$" "Preparing training process" "${EXTRA_START}"

echo "$$" >"${PID_FILE}"

declare -a CMD
CMD=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  "--code" "${CODE}"
  "--episodes-per-snr" "${EPISODES_PER_SNR}"
  "--checkpoint-dir" "${CHECKPOINT_DIR}"
  "--checkpoint-every-episodes" "${CHECKPOINT_EVERY}"
  "--log-every" "${LOG_EVERY}"
  "--seed" "${SEED}"
)

if [[ "${MAX_EPISODES}" != "null" ]]; then
  CMD+=("--max-episodes" "${MAX_EPISODES}")
fi

if [[ -f "${LATEST_CKPT}" ]]; then
  CMD+=("--resume" "${LATEST_CKPT}")
fi

printf '[%s] train command: %q\n' "$(now_iso)" "${CMD[*]}" >>"${LOG_FILE}"

EXTRA_RUNNING='{"component":"train"}'
write_state_json "${STATE_FILE}" "running" "${CODE}" "$$" "Training process started" "${EXTRA_RUNNING}"

set +e
"${CMD[@]}" >>"${LOG_FILE}" 2>&1
RC=$?
set -e

if [[ ${RC} -eq 0 ]]; then
  EXTRA_DONE=$(printf '{"component":"train","exit_code":0,"checkpoint_latest":"%s"}' "${LATEST_CKPT}")
  write_state_json "${STATE_FILE}" "done" "${CODE}" "$$" "Training completed" "${EXTRA_DONE}"
else
  EXTRA_FAIL=$(printf '{"component":"train","exit_code":%d,"checkpoint_latest":"%s"}' "${RC}" "${LATEST_CKPT}")
  write_state_json "${STATE_FILE}" "failed" "${CODE}" "$$" "Training failed" "${EXTRA_FAIL}"
fi

exit ${RC}
