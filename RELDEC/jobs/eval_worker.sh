#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <run_root> <code> [mode]" >&2
  exit 2
fi

RUN_ROOT="$1"
CODE="$2"
MODE="${3:-manual}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

MANIFEST="${RUN_ROOT}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

ensure_run_dirs "${RUN_ROOT}"

EVAL_SCRIPT="${RELDEC_DIR}/evaluate_reldec.py"
CHECKPOINT_DIR="${RUN_ROOT}/${CODE}/checkpoints"
RESULT_DIR="${RUN_ROOT}/${CODE}/results"
LOG_FILE="${RUN_ROOT}/logs/eval_${CODE}.log"
STATE_FILE="${RUN_ROOT}/state/eval_${CODE}.json"
PID_FILE="${RUN_ROOT}/pids/eval_${CODE}.pid"
mkdir -p "${RESULT_DIR}"

TARGET_FE="$(read_manifest_field "${MANIFEST}" "eval.target_frame_errors")"
MAX_FRAMES="$(read_manifest_field "${MANIFEST}" "eval.max_frames")"
SEED_BASE="$(read_manifest_field "${MANIFEST}" "eval.seed_base")"
CODE_INDEX="$(read_manifest_field "${MANIFEST}" "code_index.${CODE}")"
EVAL_SEED="$((SEED_BASE + CODE_INDEX))"

Q_TABLE_FINAL="${CHECKPOINT_DIR}/q_table_final.npy"
Q_TABLE_LATEST="${CHECKPOINT_DIR}/checkpoint_latest.npz"

if [[ -f "${Q_TABLE_FINAL}" ]]; then
  Q_TABLE_PATH="${Q_TABLE_FINAL}"
elif [[ -f "${Q_TABLE_LATEST}" ]]; then
  Q_TABLE_PATH="${Q_TABLE_LATEST}"
else
  EXTRA_WAIT='{"component":"eval"}'
  write_state_json "${STATE_FILE}" "waiting" "${CODE}" "$$" "No checkpoint available yet" "${EXTRA_WAIT}"
  echo "No Q-table/checkpoint available for ${CODE}. Skipping eval." >>"${LOG_FILE}"
  exit 3
fi

TS="$(date -u +%Y%m%d_%H%M%S)"

BERFER_CSV="${RESULT_DIR}/eval_${MODE}_berfer_${TS}.csv"
BERFER_JSON="${RESULT_DIR}/eval_${MODE}_berfer_${TS}.json"
MSG_CSV="${RESULT_DIR}/eval_${MODE}_messages_${TS}.csv"
MSG_JSON="${RESULT_DIR}/eval_${MODE}_messages_${TS}.json"

BERFER_SNR_JSON="$(read_manifest_field "${MANIFEST}" "eval.snr_berfer")"
MSG_SNR_JSON="$(read_manifest_field "${MANIFEST}" "eval.snr_messages")"
METHODS_JSON="$(read_manifest_field "${MANIFEST}" "methods")"

mapfile -t BERFER_SNR < <("${PYTHON_BIN}" - "$BERFER_SNR_JSON" <<'PY'
import json
import sys
for x in json.loads(sys.argv[1]):
    print(x)
PY
)

mapfile -t MSG_SNR < <("${PYTHON_BIN}" - "$MSG_SNR_JSON" <<'PY'
import json
import sys
for x in json.loads(sys.argv[1]):
    print(x)
PY
)

mapfile -t METHODS < <("${PYTHON_BIN}" - "$METHODS_JSON" <<'PY'
import json
import sys
for x in json.loads(sys.argv[1]):
    print(x)
PY
)

EXTRA_START=$(printf '{"component":"eval","mode":"%s","q_table":"%s"}' "${MODE}" "${Q_TABLE_PATH}")
write_state_json "${STATE_FILE}" "running" "${CODE}" "$$" "Evaluation started" "${EXTRA_START}"

echo "$$" >"${PID_FILE}"

echo "[$(now_iso)] eval start code=${CODE} mode=${MODE} q_table=${Q_TABLE_PATH}" >>"${LOG_FILE}"

init_aggregate_json() {
  local out_json="$1"
  local out_csv="$2"
  shift 2
  local snr_values=("$@")

  "${PYTHON_BIN}" - "$out_json" "$out_csv" "$CODE" "$TARGET_FE" "$MAX_FRAMES" "${snr_values[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
code = sys.argv[3]
target_fe = int(sys.argv[4])
max_frames = int(sys.argv[5])
snr_values = [float(x) for x in sys.argv[6:]]

payload = {
    "config": {
        "code": code,
        "target_frame_errors": target_fe,
        "max_frames": max_frames,
        "snr_db": snr_values,
    },
    "results": [],
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
out_csv.write_text("", encoding="utf-8")
PY
}

merge_point_json_into_aggregate() {
  local aggregate_json="$1"
  local aggregate_csv="$2"
  local point_json="$3"

  "${PYTHON_BIN}" - "$aggregate_json" "$aggregate_csv" "$point_json" <<'PY'
import csv
import json
import sys
from pathlib import Path

aggregate_json = Path(sys.argv[1])
aggregate_csv = Path(sys.argv[2])
point_json = Path(sys.argv[3])

agg = json.loads(aggregate_json.read_text(encoding="utf-8"))
point = json.loads(point_json.read_text(encoding="utf-8"))

rows = list(agg.get("results", []))
index = {(str(r.get("method", "")).lower(), float(r.get("snr_db", 0.0))): i for i, r in enumerate(rows)}

for row in point.get("results", []):
    key = (str(row.get("method", "")).lower(), float(row.get("snr_db", 0.0)))
    if key in index:
        rows[index[key]] = row
    else:
        index[key] = len(rows)
        rows.append(row)

rows.sort(key=lambda r: (float(r.get("snr_db", 0.0)), str(r.get("method", "")).lower()))
agg["results"] = rows
aggregate_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

if rows:
    fieldnames = list(rows[0].keys())
    with open(aggregate_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
else:
    aggregate_csv.write_text("", encoding="utf-8")
PY
}

run_eval_point() {
  local pass_name="$1"
  local snr="$2"
  local out_csv="$3"
  local out_json="$4"
  local seed="$5"

  local -a cmd
  cmd=(
    "${PYTHON_BIN}" "${EVAL_SCRIPT}"
    "--code" "${CODE}"
    "--q-table" "${Q_TABLE_PATH}"
    "--methods" "${METHODS[@]}"
    "--target-frame-errors" "${TARGET_FE}"
    "--max-frames" "${MAX_FRAMES}"
    "--seed" "${seed}"
    "--snr-db" "${snr}"
    "--output-csv" "${out_csv}"
    "--output-json" "${out_json}"
  )

  printf '[%s] eval %s command: %q\n' "$(now_iso)" "${pass_name}" "${cmd[*]}" >>"${LOG_FILE}"
  "${cmd[@]}" >>"${LOG_FILE}" 2>&1
}

set +e
init_aggregate_json "${BERFER_JSON}" "${BERFER_CSV}" "${BERFER_SNR[@]}"
init_aggregate_json "${MSG_JSON}" "${MSG_CSV}" "${MSG_SNR[@]}"

RC1=0
berfer_done=0
berfer_total=${#BERFER_SNR[@]}
for snr in "${BERFER_SNR[@]}"; do
  POINT_CSV="${RESULT_DIR}/tmp_${MODE}_berfer_${snr}_${TS}.csv"
  POINT_JSON="${RESULT_DIR}/tmp_${MODE}_berfer_${snr}_${TS}.json"
  point_seed=$((EVAL_SEED + berfer_done))
  run_eval_point "berfer_snr_${snr}" "${snr}" "${POINT_CSV}" "${POINT_JSON}" "${point_seed}"
  rc_point=$?
  if [[ ${rc_point} -ne 0 ]]; then
    RC1=${rc_point}
    break
  fi
  merge_point_json_into_aggregate "${BERFER_JSON}" "${BERFER_CSV}" "${POINT_JSON}"
  cp -f "${BERFER_CSV}" "${RESULT_DIR}/latest_berfer.csv"
  cp -f "${BERFER_JSON}" "${RESULT_DIR}/latest_berfer.json"
  berfer_done=$((berfer_done + 1))
  EXTRA_PROGRESS=$(printf '{"component":"eval","mode":"%s","phase":"berfer","snr_done":%d,"snr_total":%d}' "${MODE}" "${berfer_done}" "${berfer_total}")
  write_state_json "${STATE_FILE}" "running" "${CODE}" "$$" "Eval BER/FER progress" "${EXTRA_PROGRESS}"
done

RC2=0
msg_done=0
msg_total=${#MSG_SNR[@]}
if [[ ${RC1} -eq 0 ]]; then
  for snr in "${MSG_SNR[@]}"; do
    POINT_CSV="${RESULT_DIR}/tmp_${MODE}_messages_${snr}_${TS}.csv"
    POINT_JSON="${RESULT_DIR}/tmp_${MODE}_messages_${snr}_${TS}.json"
    point_seed=$((EVAL_SEED + 1000 + msg_done))
    run_eval_point "messages_snr_${snr}" "${snr}" "${POINT_CSV}" "${POINT_JSON}" "${point_seed}"
    rc_point=$?
    if [[ ${rc_point} -ne 0 ]]; then
      RC2=${rc_point}
      break
    fi
    merge_point_json_into_aggregate "${MSG_JSON}" "${MSG_CSV}" "${POINT_JSON}"
    cp -f "${MSG_CSV}" "${RESULT_DIR}/latest_messages.csv"
    cp -f "${MSG_JSON}" "${RESULT_DIR}/latest_messages.json"
    msg_done=$((msg_done + 1))
    EXTRA_PROGRESS=$(printf '{"component":"eval","mode":"%s","phase":"messages","snr_done":%d,"snr_total":%d}' "${MODE}" "${msg_done}" "${msg_total}")
    write_state_json "${STATE_FILE}" "running" "${CODE}" "$$" "Eval message progress" "${EXTRA_PROGRESS}"
  done
else
  RC2=1
fi
set -e

RC=$((RC1 != 0 ? RC1 : RC2))

if [[ ${RC} -eq 0 ]]; then
  EXTRA_DONE=$(printf '{"component":"eval","mode":"%s","exit_code":0,"latest_berfer_json":"%s","latest_messages_json":"%s"}' "${MODE}" "${RESULT_DIR}/latest_berfer.json" "${RESULT_DIR}/latest_messages.json")
  write_state_json "${STATE_FILE}" "done" "${CODE}" "$$" "Evaluation completed" "${EXTRA_DONE}"
else
  EXTRA_FAIL=$(printf '{"component":"eval","mode":"%s","exit_code":%d}' "${MODE}" "${RC}")
  write_state_json "${STATE_FILE}" "failed" "${CODE}" "$$" "Evaluation failed" "${EXTRA_FAIL}"
fi

exit ${RC}
