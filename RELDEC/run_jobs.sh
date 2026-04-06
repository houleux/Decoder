#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS_DIR="${SCRIPT_DIR}/jobs"
# shellcheck disable=SC1091
source "${JOBS_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  RELDEC/run_jobs.sh submit [--run-root PATH] [--episodes-per-snr N] [--checkpoint-every N] [--log-every N] [--max-episodes N] [--target-frame-errors N] [--max-frames N] [--interval-sec N] [--dry-run]
  RELDEC/run_jobs.sh status [--run-root PATH] [--json]
  RELDEC/run_jobs.sh eval-now [--run-root PATH] [--mode MODE]
  RELDEC/run_jobs.sh plot [--run-root PATH]
  RELDEC/run_jobs.sh restart [--run-root PATH] [--code ab|wran|all]

Notes:
  - submit launches independent training jobs and a periodic evaluator loop.
  - periodic eval stores tables only (CSV/JSON); no plotting.
  - plot is on-demand and reads latest stored tables.
EOF
}

latest_run_root() {
  local base="${SCRIPT_DIR}/notebook_runs/job_runs"
  mkdir -p "${base}"
  local latest
  latest="$(ls -1dt "${base}"/run_* 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "No run found under ${base}" >&2
    exit 2
  fi
  echo "${latest}"
}

cmd_submit() {
  local run_root=""
  local episodes_per_snr=2500
  local checkpoint_every=250
  local log_every=100
  local max_episodes="null"
  local target_fe=300
  local max_frames=200000
  local interval_sec=900
  local dry_run=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root) run_root="$2"; shift 2 ;;
      --episodes-per-snr) episodes_per_snr="$2"; shift 2 ;;
      --checkpoint-every) checkpoint_every="$2"; shift 2 ;;
      --log-every) log_every="$2"; shift 2 ;;
      --max-episodes) max_episodes="$2"; shift 2 ;;
      --target-frame-errors) target_fe="$2"; shift 2 ;;
      --max-frames) max_frames="$2"; shift 2 ;;
      --interval-sec) interval_sec="$2"; shift 2 ;;
      --dry-run) dry_run=1; shift ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "${run_root}" ]]; then
    run_root="${SCRIPT_DIR}/notebook_runs/job_runs/run_$(date -u +%Y%m%d_%H%M%S)"
  fi

  ensure_run_dirs "${run_root}"

  local cpu_count
  cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
  local worker_cap=2
  if [[ "${cpu_count}" -lt 2 ]]; then
    worker_cap=1
  fi

  local manifest="${run_root}/manifest.json"
  "${PYTHON_BIN}" - "$manifest" "$run_root" "$episodes_per_snr" "$checkpoint_every" "$log_every" "$max_episodes" "$target_fe" "$max_frames" "$interval_sec" "$cpu_count" "$worker_cap" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
run_root = sys.argv[2]
episodes_per_snr = int(sys.argv[3])
checkpoint_every = int(sys.argv[4])
log_every = int(sys.argv[5])
max_episodes_raw = sys.argv[6]
max_episodes = None if max_episodes_raw == "null" else int(max_episodes_raw)
target_fe = int(sys.argv[7])
max_frames = int(sys.argv[8])
interval_sec = int(sys.argv[9])
cpu_count = int(sys.argv[10])
worker_cap = int(sys.argv[11])

codes = ["ab", "wran"]
methods = ["flooding", "random", "round_robin", "reldec"]

manifest = {
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_root": run_root,
    "codes": codes,
    "code_index": {"ab": 0, "wran": 1},
    "methods": methods,
    "train": {
        "episodes_per_snr": episodes_per_snr,
        "checkpoint_every_episodes": checkpoint_every,
        "log_every": log_every,
        "max_episodes": max_episodes,
        "seeds": {"ab": 17, "wran": 18},
        "train_snr_count": 6,
    },
    "eval": {
        "target_frame_errors": target_fe,
        "max_frames": max_frames,
        "seed_base": 1017,
        "interval_sec": interval_sec,
        "snr_berfer": [2.0, 2.5, 3.0, 3.25, 3.5],
        "snr_messages": [2.0, 2.5, 3.0, 4.0, 5.0],
    },
    "system": {
        "cpu_count": cpu_count,
        "worker_cap": worker_cap,
    },
}

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY

  echo "Run root: ${run_root}"
  echo "Manifest: ${manifest}"

  if [[ ${dry_run} -eq 1 ]]; then
    echo "Dry run enabled; not launching jobs."
    return 0
  fi

  for code in ab wran; do
    local launch_log="${run_root}/logs/launcher_train_${code}.log"
    nohup "${JOBS_DIR}/train_worker.sh" "${run_root}" "${code}" >"${launch_log}" 2>&1 &
    echo "$!" >"${run_root}/pids/launcher_train_${code}.pid"
    echo "Launched train worker for ${code} (pid=$!)"
  done

  local periodic_log="${run_root}/logs/launcher_periodic_eval.log"
  nohup "${JOBS_DIR}/periodic_eval.sh" "${run_root}" "${interval_sec}" >"${periodic_log}" 2>&1 &
  echo "$!" >"${run_root}/pids/launcher_periodic_eval.pid"
  echo "Launched periodic evaluator (pid=$!)"
}

cmd_status() {
  local run_root=""
  local as_json=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root) run_root="$2"; shift 2 ;;
      --json) as_json=1; shift ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "${run_root}" ]]; then
    run_root="$(latest_run_root)"
  fi

  if [[ ${as_json} -eq 1 ]]; then
    "${PYTHON_BIN}" "${JOBS_DIR}/status.py" --run-root "${run_root}" --json
  else
    "${PYTHON_BIN}" "${JOBS_DIR}/status.py" --run-root "${run_root}"
  fi
}

cmd_eval_now() {
  local run_root=""
  local mode="manual"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root) run_root="$2"; shift 2 ;;
      --mode) mode="$2"; shift 2 ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "${run_root}" ]]; then
    run_root="$(latest_run_root)"
  fi

  local pids=()
  for code in ab wran; do
    "${JOBS_DIR}/eval_worker.sh" "${run_root}" "${code}" "${mode}" &
    pids+=("$!")
    echo "Started eval for ${code} (pid=$!)"
  done

  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  return ${rc}
}

cmd_plot() {
  local run_root=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root) run_root="$2"; shift 2 ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "${run_root}" ]]; then
    run_root="$(latest_run_root)"
  fi

  "${PYTHON_BIN}" "${JOBS_DIR}/plot_results.py" --run-root "${run_root}"
}

cmd_restart() {
  local run_root=""
  local code="all"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-root) run_root="$2"; shift 2 ;;
      --code) code="$2"; shift 2 ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
  done

  if [[ -z "${run_root}" ]]; then
    run_root="$(latest_run_root)"
  fi

  restart_one() {
    local c="$1"
    local pid_file="${run_root}/pids/train_${c}.pid"
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if is_pid_alive "${pid}"; then
        echo "Train job for ${c} already running (pid=${pid}); skip restart."
        return 0
      fi
    fi

    local launch_log="${run_root}/logs/launcher_train_${c}_restart.log"
    nohup "${JOBS_DIR}/train_worker.sh" "${run_root}" "${c}" >"${launch_log}" 2>&1 &
    echo "$!" >"${run_root}/pids/launcher_train_${c}.pid"
    echo "Restarted train worker for ${c} (pid=$!)"
  }

  if [[ "${code}" == "all" ]]; then
    restart_one ab
    restart_one wran
  elif [[ "${code}" == "ab" || "${code}" == "wran" ]]; then
    restart_one "${code}"
  else
    echo "Invalid --code value: ${code}" >&2
    exit 2
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 2
  fi

  local cmd="$1"
  shift

  case "${cmd}" in
    submit) cmd_submit "$@" ;;
    status) cmd_status "$@" ;;
    eval-now) cmd_eval_now "$@" ;;
    plot) cmd_plot "$@" ;;
    restart) cmd_restart "$@" ;;
    -h|--help|help) usage ;;
    *) echo "Unknown command: ${cmd}" >&2; usage; exit 2 ;;
  esac
}

main "$@"
