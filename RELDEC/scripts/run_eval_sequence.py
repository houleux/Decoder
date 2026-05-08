#!/usr/bin/env python3
"""Run training (when needed) and evaluation for multiple methods per SNR.

Behavior:
- For each SNR in the primary list, iterate methods one-by-one and run evaluation for that SNR.
- If a method requires a checkpoint and none exists, attempt a short training run before evaluation.
- Write per-run CSVs into a results directory and merge into an accumulated CSV.

This script is intended to be invoked from SLURM sbatch scripts.
"""
import argparse
import subprocess
import csv
import re
import numpy as np
from pathlib import Path
import sys
import time


def run_cmd(cmd, cwd=None):
    print("CMD:", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def _parse_method(method: str) -> tuple[str, int | None]:
    m = re.match(r"^(deep_reldec|mi_dqn|mi_tabular|mi_naive)_z(\d+)$", method)
    if not m:
        return method, None
    return m.group(1), int(m.group(2))


def ensure_checkpoint_for_method(base_dir: Path, method: str, z: int, matrix_csv: str, code: str):
    # heuristic checkpoint locations used in repo
    base = Path(base_dir) / "notebook_runs" / "continuous_reldec" / "active_run" / code / "checkpoints"
    if method.startswith("mi_dqn"):
        ck = base / f"checkpoints_mi_dqn_z{z}_1k" / "checkpoint_latest.npz"
    elif method.startswith("deep_reldec_z"):
        ck = base / f"checkpoints_deep_z{z}_1k" / "checkpoint_latest.npz"
    elif method.startswith("mi_tabular_z"):
        ck = base / f"checkpoints_mi_tabular_z{z}_1k" / "checkpoint_latest.npz"
    else:
        ck = None

    if ck is None:
        return None
    if ck.exists():
        print(f"Found checkpoint for {method}: {ck}")
        return ck

    # attempt a short training run to produce a checkpoint
    print(f"No checkpoint found for {method}, attempting short training to produce one: {ck}")
    # conservative short training call; only map to supported policy types
    policy = None
    if method.startswith("mi_dqn"):
        policy = f"mi_dqn_z{int(z)}"
    elif method.startswith("deep_reldec"):
        policy = f"deep_z{int(z)}"
    elif method.startswith("mi_tabular"):
        policy = f"mi_tabular_z{int(z)}"

    if policy is None:
        print(f"Auto-training not supported for method={method} with z={z}; skipping checkpoint creation")
        return None

    train_cmd = (
        f"python3 RELDEC/train_reldec.py --code {args.code} --matrix-csv {matrix_csv} "
        f"--policy-type {policy} --device cpu --resume '' --checkpoint-dir {ck.parent} "
        f"--checkpoint-every-episodes 250 --log-every 100 --max-episodes 250"
    )
    run_cmd(train_cmd, cwd=str(Path.cwd()))
    if ck.exists():
        return ck
    return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-csv", required=True)
    p.add_argument("--code", choices=["ab","wran"], required=True)
    p.add_argument("--z", type=int, required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--snr-primary", nargs="+", type=float, default=[0.5, 1.5, 2.5])
    p.add_argument("--snr-secondary", nargs="+", type=float, default=[1.0, 2.0])
    p.add_argument("--result-dir", default="RELDEC/notebook_runs/continuous_reldec/active_run/custom/results")
    p.add_argument("--q-table", default=None)
    p.add_argument("--target-frame-errors", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=10000)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Locate a q-table if available (used by tabular 'reldec' evaluations)
    q_table_candidates = list(Path('RELDEC').rglob('q_table_final.npy'))
    q_table_arg_global = ''
    if args.q_table:
        q_table_arg_global = f"--q-table {args.q_table}"
    elif q_table_candidates:
        # prefer a q-table next to result dir, then fallback to shape match
        preferred = result_dir.parent / "checkpoints_reldec_tabular" / "q_table_final.npy"
        if preferred.exists():
            q_table_arg_global = f"--q-table {preferred}"
        else:
            # compute expected q-table shape from provided matrix
            matrix_path = Path(args.matrix_csv)
            rows = {}
            with open(matrix_path, newline='') as fh:
                rdr = csv.reader(fh)
                header = next(rdr, None)
                for r in rdr:
                    if not r:
                        continue
                    row_idx = int(r[0])
                    rows.setdefault(row_idx, 0)
                    rows[row_idx] += 1
            if rows:
                m = max(rows.keys()) + 1
                max_degree = max(rows.values())
                expected_shape = (1 << max_degree, m)
            else:
                expected_shape = None

            chosen = None
            for q in q_table_candidates:
                try:
                    arr = np.load(q)
                except Exception:
                    continue
                if expected_shape is not None and arr.shape == expected_shape:
                    chosen = q
                    break
            if chosen is None:
                chosen = q_table_candidates[0]
                print('No q-table matched expected shape; falling back to', chosen)
            else:
                print('Selected q-table matching expected shape:', chosen)

            q_table_arg_global = f'--q-table {chosen}'
    else:
        print('No q-table found; reldec evaluations will be skipped unless a q-table is provided')

    primary = list(args.snr_primary)
    secondary = list(args.snr_secondary)

    def eval_for_snr_list(snr_list, pass_name):
        for snr in snr_list:
            for method in args.methods:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_csv = result_dir / f"eval_{method}_snr{snr:.1f}_{pass_name}_{ts}.csv"

                # If method needs checkpoint, ensure it exists (train if missing)
                deep_arg = ""
                mi_tabular_arg = ""
                base, method_z = _parse_method(method)
                z = method_z if method_z is not None else args.z
                if base in {"mi_dqn", "deep_reldec", "mi_tabular"}:
                    ck = ensure_checkpoint_for_method(Path('.'), method, z, args.matrix_csv, args.code)
                    if ck and base in {"mi_dqn", "deep_reldec"}:
                        deep_arg = f"--deep-checkpoint {ck}"
                    if ck and base == "mi_tabular":
                        q_table = ck.parent / "q_table_final.npy"
                        mi_tabular_arg = f"--mi-tabular-q-table {q_table}"

                # If method is 'reldec' and we have a global q-table, include it; otherwise skip reldec
                if method == 'reldec' and not q_table_arg_global:
                    print(f"Skipping method 'reldec' for snr={snr} because no q-table is available")
                    continue

                cmd = (
                    f"python3 RELDEC/evaluate_reldec.py --code {args.code} --matrix-csv {args.matrix_csv} "
                    f"{q_table_arg_global} {mi_tabular_arg} --methods {method} {deep_arg} "
                    f"--snr-db {snr} --target-frame-errors {args.target_frame_errors} --max-frames {args.max_frames} "
                    f"--seed {args.seed} --output-csv {out_csv}"
                )

                # run evaluation single-SNR single-method
                run_cmd(cmd, cwd=str(Path.cwd()))

    # Run primary pass
    eval_for_snr_list(primary, "primary")

    # After primary pass finishes, run secondary
    eval_for_snr_list(secondary, "secondary")

    # Merge results into accumulated CSV
    merge_py = f"""
from pathlib import Path
import pandas as pd
rd=Path('{result_dir}')
acc=rd/'accumulated_results.csv'
frames=[]
if acc.exists():
    frames.append(pd.read_csv(acc))
for p in sorted(rd.glob('eval_*.csv')):
    frames.append(pd.read_csv(p))
if frames:
    df=pd.concat(frames,ignore_index=True)
    key_cols=['method','snr_db','code','matrix_csv','target_frame_errors','max_frames','all_zero_only']
    df=df.drop_duplicates(subset=key_cols,keep='last')
    df=df.sort_values(['snr_db','method']).reset_index(drop=True)
    df.to_csv(acc,index=False)
    print('wrote', acc)
"""

    run_cmd(f"python3 - <<'PY'\n{merge_py}\nPY")
