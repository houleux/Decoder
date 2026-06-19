#!/usr/bin/env bash
# eval_all_parallel.sh
# Runs all trained WRAN evaluations in parallel (reldec, reldec_misq_global,
# tabular_augmented_max_avg_zx) across z in {1,2,4,6} and shows a live
# per-job progress bar.
#
# Usage:
#   bash eval_all_parallel.sh [--max-frames N]
#
# Defaults: --max-frames 100, --target-frame-errors 300, --snr-db 1.0 2.0 3.0 4.0

set -euo pipefail

# ── configurable ──────────────────────────────────────────────────────────────
MAX_FRAMES="${MAX_FRAMES:-100}"
TARGET_FE="${TARGET_FE:-300}"
SEED=42
SNR_DB="1.0 2.0 3.0 4.0"
export PYTHONPATH=".:ldpc/src_python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/RELDEC/logs/eval_parallel"
RESULTS_DIR="${SCRIPT_DIR}/RELDEC/results"
CKPT_DIR="${SCRIPT_DIR}/RELDEC/checkpoints/0613_013444"
LIVE_CSV="${RESULTS_DIR}/eval_all_live.csv"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── build job list ────────────────────────────────────────────────────────────
# Each entry: "label|cmd"
declare -a JOBS=() JOB_LOGS=() JOB_ZVALS=()

for Z in 1 2 4 6; do
  # reldec
  JOBS+=("reldec_z${Z}|PYTHONPATH=\".:ldpc/src_python\" python3 -m RELDEC.evaluate_reldec \
--code wran \
--methods reldec \
--snr-db ${SNR_DB} \
--i-max 50 \
--target-frame-errors ${TARGET_FE} \
--max-frames ${MAX_FRAMES} \
--seed ${SEED} \
--z ${Z} \
--q-table ${CKPT_DIR}_reldec_wran_z${Z}/q_table_final.npy \
--output-csv ${RESULTS_DIR}/eval_reldec_z${Z}.csv")
  JOB_LOGS+=("${LOG_DIR}/reldec_z${Z}.log"); JOB_ZVALS+=("$Z")

  # reldec_misq_global
  JOBS+=("reldec_misq_global_z${Z}|PYTHONPATH=\".:ldpc/src_python\" python3 -m RELDEC.evaluate_reldec \
--code wran \
--methods reldec_misq_global \
--snr-db ${SNR_DB} \
--i-max 50 \
--target-frame-errors ${TARGET_FE} \
--max-frames ${MAX_FRAMES} \
--seed ${SEED} \
--z ${Z} \
--q-table ${CKPT_DIR}_reldec_misq_global_wran_z${Z}/q_table_final.npy \
--output-csv ${RESULTS_DIR}/eval_reldec_misq_global_z${Z}.csv")
  JOB_LOGS+=("${LOG_DIR}/reldec_misq_global_z${Z}.log"); JOB_ZVALS+=("$Z")

  # tabular_augmented_max_avg_zx
  JOBS+=("tabular_aug_z${Z}|PYTHONPATH=\".:ldpc/src_python\" python3 -m RELDEC.evaluate_reldec \
--code wran \
--methods tabular_augmented_max_avg_zx \
--snr-db ${SNR_DB} \
--i-max 50 \
--target-frame-errors ${TARGET_FE} \
--max-frames ${MAX_FRAMES} \
--seed ${SEED} \
--z ${Z} \
--tabular-augmented-q-table ${CKPT_DIR}_tabular_augmented_max_avg_zx_wran_z${Z}/q_table_final.npy \
--output-csv ${RESULTS_DIR}/eval_tabular_augmented_max_avg_zx_z${Z}.csv")
  JOB_LOGS+=("${LOG_DIR}/tabular_aug_z${Z}.log"); JOB_ZVALS+=("$Z")
done

TOTAL=${#JOBS[@]}

# ── state arrays (indexed by job index) ──────────────────────────────────────
declare -a LABELS PID_ARR STATUS START_TIME

for i in "${!JOBS[@]}"; do
  IFS='|' read -r lbl _ <<< "${JOBS[$i]}"
  LABELS[$i]="$lbl"
  STATUS[$i]="PENDING"
  PID_ARR[$i]=""
  START_TIME[$i]=""
done

# ── launch all jobs ───────────────────────────────────────────────────────────
for i in "${!JOBS[@]}"; do
  IFS='|' read -r lbl cmd <<< "${JOBS[$i]}"
  LOG="${LOG_DIR}/${lbl}.log"
  eval "$cmd" >"$LOG" 2>&1 &
  PID_ARR[$i]=$!
  STATUS[$i]="RUNNING"
  START_TIME[$i]=$(date +%s)
done

# ── live CSV aggregator (parses log output per SNR point as it completes) ──────
update_live_csv() {
  local tmp="${LIVE_CSV}.tmp"
  echo 'z,method,snr_db,frames,fer,ber,avg_messages' > "$tmp"
  for i in "${!JOB_LOGS[@]}"; do
    local log="${JOB_LOGS[$i]}"
    local z="${JOB_ZVALS[$i]}"
    [[ ! -f "$log" ]] && continue
    # The eval script prints one result line per SNR point as it runs:
    #   [eval] snr=X.XX dB
    #   - METHOD    frames=      N FER=X BER=Y avg_msgs=Z
    # Parse both with awk to emit CSV rows without waiting for job completion.
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

# ── draw progress board ───────────────────────────────────────────────────────
draw_board() {
  local done_count=0
  local failed_count=0
  printf "\033[2J\033[H"   # clear screen + home
  printf "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}\n"
  printf "${BOLD}  WRAN Parallel Evaluation  |  %d jobs${RESET}\n" "$TOTAL"
  printf "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}\n"

  for i in "${!JOBS[@]}"; do
    local lbl="${LABELS[$i]}"
    local st="${STATUS[$i]}"
    local pid="${PID_ARR[$i]}"
    local elapsed=""

    if [[ -n "${START_TIME[$i]}" ]]; then
      local now; now=$(date +%s)
      local secs=$(( now - START_TIME[$i] ))
      elapsed=$(printf "%dm%02ds" $((secs/60)) $((secs%60)))
    fi

    # Check if process finished
    if [[ "$st" == "RUNNING" ]]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null
        local rc=$?
        if [[ $rc -eq 0 ]]; then
          STATUS[$i]="DONE"
          st="DONE"
        else
          STATUS[$i]="FAILED"
          st="FAILED"
        fi
      fi
    fi

    case "$st" in
      RUNNING) printf "  ${YELLOW}⏳ %-32s  %s${RESET}\n" "$lbl" "$elapsed" ;;
      DONE)    printf "  ${GREEN}✅ %-32s  %s${RESET}\n"  "$lbl" "$elapsed" ; (( done_count++ )) ;;
      FAILED)  printf "  ${RED}❌ %-32s  %s${RESET}\n"   "$lbl" "$elapsed" ; (( failed_count++ )) ;;
      PENDING) printf "  ⬜ %-32s\n" "$lbl" ;;
    esac
  done

  # Overall progress bar
  local fin=$(( done_count + failed_count ))
  local pct=$(( fin * 100 / TOTAL ))
  local bar_width=50
  local filled=$(( pct * bar_width / 100 ))
  local bar=""
  for (( b=0; b<filled; b++ ));  do bar+="█"; done
  for (( b=filled; b<bar_width; b++ )); do bar+="░"; done

  printf "\n  ${BOLD}Progress: [${GREEN}%s${RESET}${BOLD}] %d%%  (%d/%d done)${RESET}\n" \
    "$bar" "$pct" "$fin" "$TOTAL"

  if [[ "$failed_count" -gt 0 ]]; then
    printf "  ${RED}${BOLD}%d job(s) failed — check logs in %s${RESET}\n" "$failed_count" "$LOG_DIR"
  fi

  # Return 1 if all done
  [[ $fin -eq $TOTAL ]] && return 1 || return 0
}

# ── poll loop ─────────────────────────────────────────────────────────────────
while draw_board; do
  update_live_csv
  sleep 2
done

# Final render + final CSV flush
draw_board || true
update_live_csv

printf "\n${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}\n"
printf "${BOLD}All jobs finished.  Results in: ${RESULTS_DIR}${RESET}\n"
printf "${BOLD}Live CSV: ${LIVE_CSV}${RESET}\n"

# Summary of failed jobs
FAILED_JOBS=()
for i in "${!JOBS[@]}"; do
  if [[ "${STATUS[$i]}" == "FAILED" ]]; then
    FAILED_JOBS+=("${LABELS[$i]}")
    printf "${RED}  FAILED: %s  →  log: %s/%s.log${RESET}\n" \
      "${LABELS[$i]}" "$LOG_DIR" "${LABELS[$i]}"
  fi
done

[[ ${#FAILED_JOBS[@]} -gt 0 ]] && exit 1 || exit 0
