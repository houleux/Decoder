#!/usr/bin/env bash
# eval_mi_parallel.sh
# Trains + evaluates mi_tabular_zx (z=1,2,4,6) and evaluates mi_naive_zx (z=1,2,4,6)
# All 8 jobs run in parallel; mi_tabular jobs run train → eval sequentially
# within their own sub-shell while the progress bar tracks their phase.
#
# Usage:
#   bash eval_mi_parallel.sh [options]
#
# Env overrides:
#   MAX_FRAMES=100           frames per SNR point at evaluation
#   TARGET_FE=300            target frame errors
#   EPS_PER_SNR=2            training episodes per SNR point (for mi_tabular)
#   MI_BINS=4                MI quantisation bins

set -uo pipefail

# ── configurable ──────────────────────────────────────────────────────────────
MAX_FRAMES="${MAX_FRAMES:-100}"
TARGET_FE="${TARGET_FE:-300}"
EPS_PER_SNR="${EPS_PER_SNR:-2}"
MI_BINS="${MI_BINS:-4}"
SEED=42
TRAIN_SNR="1.0 2.0 3.0 4.0 5.0 5.5"   # wran train schedule (unchanged)
EVAL_SNR="1.0 2.0 3.0 4.0"

export PYTHONPATH=".:ldpc/src_python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/RELDEC/logs/eval_mi_parallel"
RESULTS_DIR="${SCRIPT_DIR}/RELDEC/results"
CKPT_BASE="${SCRIPT_DIR}/RELDEC/checkpoints/mi_tabular_wran"
LIVE_CSV="${RESULTS_DIR}/eval_mi_live.csv"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

# ── job runner functions ──────────────────────────────────────────────────────

# mi_naive_zx: evaluate only (no training)
run_mi_naive() {
  local Z=$1
  local LOG=$2
  {
    echo "[phase] EVALUATING"
    PYTHONPATH=".:ldpc/src_python" python3 -m RELDEC.evaluate_reldec \
      --code wran \
      --methods mi_naive_zx \
      --snr-db ${EVAL_SNR} \
      --i-max 50 \
      --target-frame-errors "${TARGET_FE}" \
      --max-frames "${MAX_FRAMES}" \
      --seed "${SEED}" \
      --z "${Z}" \
      --output-csv "${RESULTS_DIR}/eval_mi_naive_z${Z}.csv"
    echo "[phase] DONE"
  } >"$LOG" 2>&1
}

# mi_tabular_zx: train then evaluate
run_mi_tabular() {
  local Z=$1
  local LOG=$2
  local CKPT_DIR="${CKPT_BASE}_z${Z}"
  {
    echo "[phase] TRAINING"
    PYTHONPATH=".:ldpc/src_python" python3 -m RELDEC.train_reldec \
      --code wran \
      --policy-type mi_tabular_zx \
      --z "${Z}" \
      --mi-bins "${MI_BINS}" \
      --snr-db ${TRAIN_SNR} \
      --episodes-per-snr "${EPS_PER_SNR}" \
      --seed "${SEED}" \
      --checkpoint-dir "${CKPT_DIR}"
    local train_rc=$?
    if [[ $train_rc -ne 0 ]]; then
      echo "[phase] FAILED_TRAIN"
      exit $train_rc
    fi

    echo "[phase] EVALUATING"
    PYTHONPATH=".:ldpc/src_python" python3 -m RELDEC.evaluate_reldec \
      --code wran \
      --methods mi_tabular_zx \
      --snr-db ${EVAL_SNR} \
      --i-max 50 \
      --target-frame-errors "${TARGET_FE}" \
      --max-frames "${MAX_FRAMES}" \
      --seed "${SEED}" \
      --z "${Z}" \
      --mi-bins "${MI_BINS}" \
      --mi-tabular-q-table "${CKPT_DIR}/q_table_final.npy" \
      --output-csv "${RESULTS_DIR}/eval_mi_tabular_z${Z}.csv"
    echo "[phase] DONE"
  } >"$LOG" 2>&1
}

# ── build job list ────────────────────────────────────────────────────────────
declare -a LABELS LOG_FILES TYPES JOB_CSVS JOB_ZVALS
for Z in 1 2 4 6; do
  LABELS+=("mi_naive_z${Z}")
  LOG_FILES+=("${LOG_DIR}/mi_naive_z${Z}.log")
  TYPES+=("naive")
  JOB_CSVS+=("${RESULTS_DIR}/eval_mi_naive_z${Z}.csv")
  JOB_ZVALS+=("${Z}")
done
for Z in 1 2 4 6; do
  LABELS+=("mi_tabular_z${Z}")
  LOG_FILES+=("${LOG_DIR}/mi_tabular_z${Z}.log")
  TYPES+=("tabular")
  JOB_CSVS+=("${RESULTS_DIR}/eval_mi_tabular_z${Z}.csv")
  JOB_ZVALS+=("${Z}")
done

TOTAL=${#LABELS[@]}

# ── state arrays ──────────────────────────────────────────────────────────────
declare -a PID_ARR STATUS PHASE START_TIME

for i in "${!LABELS[@]}"; do
  PID_ARR[$i]=""
  STATUS[$i]="PENDING"
  PHASE[$i]=""
  START_TIME[$i]=""
done

# ── launch all jobs ───────────────────────────────────────────────────────────
for i in "${!LABELS[@]}"; do
  local_type="${TYPES[$i]}"
  local_log="${LOG_FILES[$i]}"

  # extract Z from label (last character after underscore)
  local_z="${LABELS[$i]##*z}"

  if [[ "$local_type" == "naive" ]]; then
    run_mi_naive "$local_z" "$local_log" &
  else
    run_mi_tabular "$local_z" "$local_log" &
  fi
  PID_ARR[$i]=$!
  STATUS[$i]="RUNNING"
  PHASE[$i]="STARTING"
  START_TIME[$i]=$(date +%s)
done

# ── live CSV aggregator (parses log output per SNR point as it completes) ──────
update_live_csv() {
  local tmp="${LIVE_CSV}.tmp"
  echo 'z,method,snr_db,frames,fer,ber,avg_messages' > "$tmp"
  for i in "${!LOG_FILES[@]}"; do
    local log="${LOG_FILES[$i]}"
    local z="${JOB_ZVALS[$i]}"
    [[ ! -f "$log" ]] && continue
    # The eval script prints one result line per SNR point as it runs:
    #   [eval] snr=X.XX dB
    #   - METHOD    frames=      N FER=X BER=Y avg_msgs=Z
    # For mi_tabular logs, training output precedes eval output;
    # training lines never match [eval] snr=, so awk silently skips them.
    awk -v z="$z" '
      /^\[eval\] snr=/ {
        s = $0; sub(/.*snr=/, "", s); sub(/ dB.*/, "", s); snr = s
      }
      /^  - [a-z_]/ && snr != "" {
        method = $2; frames = $4
        fer = ""; ber = ""; avg_msgs = ""
        for (k = 5; k <= NF; k++) {
          if ($k ~ /^FER=/)      { fer    = substr($k, 5) }
          if ($k ~ /^BER=/)      { ber    = substr($k, 5) }
          if ($k ~ /^avg_msgs=/) { avg_msgs = substr($k, 10) }
        }
        if (frames != "" && fer != "" && ber != "" && avg_msgs != "")
          print z "," method "," snr "," frames "," fer "," ber "," avg_msgs
      }
    ' "$log" >> "$tmp"
  done
  mv "$tmp" "$LIVE_CSV"
}

# ── phase detection from log ──────────────────────────────────────────────────
get_phase_from_log() {
  local log=$1
  if [[ ! -f "$log" ]]; then echo "STARTING"; return; fi
  # read last [phase] line
  local p
  p=$(grep -oP '(?<=\[phase\] )\S+' "$log" 2>/dev/null | tail -1)
  echo "${p:-STARTING}"
}

# ── draw progress board ───────────────────────────────────────────────────────
draw_board() {
  local done_count=0 failed_count=0

  printf "\033[2J\033[H"
  printf "${BOLD}${CYAN}════════════════════════════════════════════════════════${RESET}\n"
  printf "${BOLD}  MI Parallel Evaluation  |  %d jobs (mi_naive + mi_tabular)${RESET}\n" "$TOTAL"
  printf "${BOLD}${CYAN}════════════════════════════════════════════════════════${RESET}\n"
  printf "  %-30s  %-12s  %s\n" "Job" "Phase" "Elapsed"
  printf "  %-30s  %-12s  %s\n" "───────────────────────────────" "────────────" "───────"

  for i in "${!LABELS[@]}"; do
    local lbl="${LABELS[$i]}"
    local st="${STATUS[$i]}"
    local pid="${PID_ARR[$i]}"
    local elapsed="—"

    if [[ -n "${START_TIME[$i]}" ]]; then
      local now; now=$(date +%s)
      local secs=$(( now - START_TIME[$i] ))
      elapsed=$(printf "%dm%02ds" $((secs/60)) $((secs%60)))
    fi

    # update phase from log
    if [[ "$st" == "RUNNING" ]]; then
      PHASE[$i]=$(get_phase_from_log "${LOG_FILES[$i]}")
    fi

    # check if process finished
    if [[ "$st" == "RUNNING" ]]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        local rc=$?
        if [[ $rc -eq 0 ]]; then
          STATUS[$i]="DONE"; st="DONE"; PHASE[$i]="DONE"
        else
          STATUS[$i]="FAILED"; st="FAILED"
        fi
      fi
    fi

    local phase_str="${PHASE[$i]:-—}"
    case "$st" in
      DONE)
        printf "  ${GREEN}✅ %-30s  %-12s  %s${RESET}\n" "$lbl" "DONE" "$elapsed"
        (( done_count++ ))
        ;;
      FAILED)
        printf "  ${RED}❌ %-30s  %-12s  %s${RESET}\n" "$lbl" "FAILED" "$elapsed"
        (( failed_count++ ))
        ;;
      RUNNING)
        case "$phase_str" in
          TRAINING)   printf "  ${YELLOW}🔧 %-30s  %-12s  %s${RESET}\n" "$lbl" "TRAINING"   "$elapsed" ;;
          EVALUATING) printf "  ${BLUE}📊 %-30s  %-12s  %s${RESET}\n"   "$lbl" "EVALUATING" "$elapsed" ;;
          *)          printf "  ${YELLOW}⏳ %-30s  %-12s  %s${RESET}\n" "$lbl" "STARTING"   "$elapsed" ;;
        esac
        ;;
      PENDING)
        printf "  ⬜ %-30s  %-12s\n" "$lbl" "PENDING"
        ;;
    esac
  done

  # overall progress bar
  local fin=$(( done_count + failed_count ))
  local pct=$(( fin * 100 / TOTAL ))
  local bar_width=50
  local filled=$(( pct * bar_width / 100 ))
  local bar=""
  for (( b=0; b<filled; b++ ));        do bar+="█"; done
  for (( b=filled; b<bar_width; b++ )); do bar+="░"; done

  printf "\n  ${BOLD}Progress: [${GREEN}%s${RESET}${BOLD}] %d%%  (%d/%d)${RESET}\n" \
    "$bar" "$pct" "$fin" "$TOTAL"

  if [[ "$failed_count" -gt 0 ]]; then
    printf "  ${RED}${BOLD}%d job(s) FAILED — logs in: %s${RESET}\n" "$failed_count" "$LOG_DIR"
  fi

  [[ $fin -eq $TOTAL ]] && return 1 || return 0
}

# ── poll loop ─────────────────────────────────────────────────────────────────
while draw_board; do
  update_live_csv
  sleep 2
done
draw_board || true
update_live_csv

printf "\n${BOLD}${CYAN}════════════════════════════════════════════════════════${RESET}\n"
printf "${BOLD}All jobs finished.  Results in: %s${RESET}\n" "$RESULTS_DIR"
printf "${BOLD}Live CSV: ${LIVE_CSV}${RESET}\n"

# list failed jobs and their logs
FAILED_ANY=0
for i in "${!LABELS[@]}"; do
  if [[ "${STATUS[$i]}" == "FAILED" ]]; then
    printf "${RED}  FAILED: %-28s  log → %s${RESET}\n" \
      "${LABELS[$i]}" "${LOG_FILES[$i]}"
    FAILED_ANY=1
  fi
done

[[ $FAILED_ANY -eq 0 ]] && exit 0 || exit 1
