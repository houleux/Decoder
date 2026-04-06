#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_root> <interval_seconds>" >&2
  exit 2
fi

RUN_ROOT="$1"
INTERVAL_SEC="$2"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MANIFEST="${RUN_ROOT}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

LOG_FILE="${RUN_ROOT}/logs/periodic_eval.log"
STATE_FILE="${RUN_ROOT}/state/periodic_eval.json"
PID_FILE="${RUN_ROOT}/pids/periodic_eval.pid"

echo "$$" >"${PID_FILE}"
write_state_json "${STATE_FILE}" "running" "fleet" "$$" "Periodic evaluator started" '{"component":"periodic_eval"}'

CODES_JSON="$(read_manifest_field "${MANIFEST}" "codes")"
mapfile -t CODES < <("${PYTHON_BIN}" - "$CODES_JSON" <<'PY'
import json
import sys
for x in json.loads(sys.argv[1]):
    print(x)
PY
)

# Give training workers a brief startup window to create PID/checkpoint files.
sleep 5

has_live_training() {
  local code
  for code in "${CODES[@]}"; do
    local pid_file="${RUN_ROOT}/pids/train_${code}.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if is_pid_alive "${pid}"; then
        return 0
      fi
    fi
  done
  return 1
}

has_any_checkpoint() {
  local code
  for code in "${CODES[@]}"; do
    if [[ -f "${RUN_ROOT}/${code}/checkpoints/checkpoint_latest.npz" ]] || [[ -f "${RUN_ROOT}/${code}/checkpoints/q_table_final.npy" ]]; then
      return 0
    fi
  done
  return 1
}

no_checkpoint_polls=0
max_no_checkpoint_polls=12

while true; do
  if has_live_training; then
    echo "[$(now_iso)] periodic eval cycle start" >>"${LOG_FILE}"
    for code in "${CODES[@]}"; do
      if "${SCRIPT_DIR}/eval_worker.sh" "${RUN_ROOT}" "${code}" "periodic" >>"${LOG_FILE}" 2>&1; then
        echo "[$(now_iso)] periodic eval ok code=${code}" >>"${LOG_FILE}"
      else
        echo "[$(now_iso)] periodic eval failed code=${code}" >>"${LOG_FILE}"
      fi
    done

    write_state_json "${STATE_FILE}" "running" "fleet" "$$" "Periodic cycle completed" '{"component":"periodic_eval"}'
    sleep "${INTERVAL_SEC}"
  else
    if has_any_checkpoint; then
      echo "[$(now_iso)] no live training jobs; running final periodic snapshot" >>"${LOG_FILE}"
      for code in "${CODES[@]}"; do
        "${SCRIPT_DIR}/eval_worker.sh" "${RUN_ROOT}" "${code}" "final" >>"${LOG_FILE}" 2>&1 || true
      done
      write_state_json "${STATE_FILE}" "done" "fleet" "$$" "Periodic evaluator finished" '{"component":"periodic_eval"}'
      exit 0
    fi

    no_checkpoint_polls=$((no_checkpoint_polls + 1))
    echo "[$(now_iso)] no live training and no checkpoints yet; waiting (${no_checkpoint_polls}/${max_no_checkpoint_polls})" >>"${LOG_FILE}"
    write_state_json "${STATE_FILE}" "running" "fleet" "$$" "Waiting for training/checkpoint startup" '{"component":"periodic_eval"}'
    if [[ ${no_checkpoint_polls} -ge ${max_no_checkpoint_polls} ]]; then
      write_state_json "${STATE_FILE}" "failed" "fleet" "$$" "Periodic evaluator timed out waiting for checkpoints" '{"component":"periodic_eval","exit_code":4}'
      exit 4
    fi
    sleep 5
  fi
done
