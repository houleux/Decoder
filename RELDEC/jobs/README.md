# RELDEC Job Framework (Bash-first)

This folder provides a low-fuss orchestration layer around existing RELDEC CLIs.

## Goals
- independent OS jobs (no notebook parallel kernels)
- crash-safe resume via checkpoint files
- table-first periodic evaluation (no periodic plotting)
- always-on status from state files + checkpoint progress
- on-demand plotting from stored results

## Commands
Run from workspace root:

```bash
bash RELDEC/run_jobs.sh submit
bash RELDEC/run_jobs.sh status
bash RELDEC/run_jobs.sh eval-now
bash RELDEC/run_jobs.sh plot
bash RELDEC/run_jobs.sh restart --code all
```

## Submit options
```bash
bash RELDEC/run_jobs.sh submit \
  --episodes-per-snr 2500 \
  --checkpoint-every 250 \
  --log-every 100 \
  --target-frame-errors 300 \
  --max-frames 200000 \
  --interval-sec 900
```

## Evaluation SNR policy
- BER/FER tables: `2.0, 2.5, 3.0, 3.25, 3.5`
- Avg CN->VN message tables: `2.0, 2.5, 3.0, 4.0, 5.0`

Periodic jobs save timestamped CSV/JSON snapshots only.

## Artifacts
For each run root under `RELDEC/notebook_runs/job_runs/run_<timestamp>`:

- `manifest.json`: immutable run config
- `state/*.json`: worker state files
- `logs/*.log`: train/eval/periodic logs
- `pids/*.pid`: worker PIDs
- `<code>/checkpoints/*`: train checkpoints
- `<code>/results/eval_*_*.csv/json`: timestamped eval snapshots
- `<code>/results/latest_berfer.*`: latest BER/FER table snapshot
- `<code>/results/latest_messages.*`: latest complexity table snapshot
- `plots/*.png`: on-demand plot output
