#!/usr/bin/env python3
"""Run partial baseline smoke (Mackay + WRAN), train+eval each method, record timings.

This script reads `RELDEC/configs/baseline_partial_smoke.yaml` and runs
training (when required) and evaluation for each method x matrix combination.
It writes a CSV `RELDEC/results/partial_smoke_timings.csv` with timings and
an estimated full-run time by scaling from smoke budget.
"""

from __future__ import annotations

import subprocess
import time
import yaml
from pathlib import Path
import csv
import os

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "baseline_partial_smoke.yaml"
RESULTS_CSV = ROOT / "results" / "partial_smoke_timings.csv"

# Map short matrix keys to csv filenames in RELDEC/matrices
MATRIX_MAP = {
    "mackay": ROOT / "matrices" / "H_Mackay_96_48.csv",
    "wran": ROOT / "matrices" / "WRAN_irreg_384_256.csv",
}

# Simple mapping from method -> training policy_type
def policy_for_method(method: str) -> str:
    if method == "reldec":
        return "tabular"
    if method.endswith("_zx") or method.startswith("deep_") or method.startswith("mi_dqn") or method.startswith("augmented_"):
        return "deep_zx"
    if method.startswith("mi_tabular"):
        return "mi_tabular_zx"
    return "tabular"


def run_cmd(cmd: list[str], env=None) -> int:
    # Ensure PYTHONPATH includes project root and ldpc extension folder
    env_vars = os.environ.copy()
    project_root = str(ROOT)
    ldpc_path = str(ROOT / "ldpc" / "src_python")
    existing = env_vars.get("PYTHONPATH", "")
    paths = [project_root, ldpc_path]
    if existing:
        paths.append(existing)
    env_vars["PYTHONPATH"] = ":".join(paths)

    start = time.time()
    proc = subprocess.run(cmd, env=env_vars)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    matrices = cfg["benchmarking"]["matrices"]
    methods = cfg["benchmarking"]["methods"]

    tabular_eps = cfg["benchmarking"]["train"]["tabular"]["episodes_per_snr"]
    deep_eps = cfg["benchmarking"]["train"]["deep"]["episodes_per_snr"]
    smoke_eps_tabular = tabular_eps
    smoke_eps_deep = deep_eps

    snr_db = cfg["benchmarking"]["evaluation"]["snr_db"]
    i_max = cfg["benchmarking"]["evaluation"].get("i_max", 20)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for matrix_key in matrices:
        matrix_csv = MATRIX_MAP.get(matrix_key)
        if matrix_csv is None or not matrix_csv.exists():
            print(f"Skipping missing matrix: {matrix_key}")
            continue

        for method in methods:
            print(f"Running: matrix={matrix_key} method={method}")
            policy = policy_for_method(method)
            # Build checkpoint dir per combo
            ckpt_dir = ROOT / "checkpoints" / f"{method}_{matrix_key}_smoke"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            train_time = 0.0
            eval_time = 0.0
            train_rc = 0
            eval_rc = 0

            # Determine if training is required by policy
            needs_training = policy in {"tabular", "mi_tabular_zx", "deep_zx", "deep_z1"}
            # Choose episodes
            eps = smoke_eps_tabular if policy in {"tabular", "mi_tabular_zx"} else smoke_eps_deep

            if needs_training:
                train_cmd = [
                    "python3",
                    "-m",
                    "RELDEC.train_reldec",
                    "--code",
                    matrix_key,
                    "--matrix-csv",
                    str(matrix_csv),
                    "--policy-type",
                    policy,
                    "--episodes-per-snr",
                    str(eps),
                    "--seed",
                    str(cfg["system"]["seed"]),
                    "--device",
                    "cuda",
                    "--checkpoint-dir",
                    str(ckpt_dir),
                    "--checkpoint-every-episodes",
                    str(max(1, int(eps // 4))),
                ]

                print("  > training:", " ".join(train_cmd))
                rc, train_time = run_cmd(train_cmd)
                train_rc = rc

            # Determine checkpoint paths for evaluation
            q_table = None
            deep_ckpt = None
            if (ckpt_dir / "q_table_final.npy").exists():
                q_table = str(ckpt_dir / "q_table_final.npy")
            if (ckpt_dir / "dqn_final.npz").exists():
                deep_ckpt = str(ckpt_dir / "dqn_final.npz")

            eval_cmd = [
                "python3",
                "-m",
                "RELDEC.evaluate_reldec",
                "--code",
                matrix_key,
                "--matrix-csv",
                str(matrix_csv),
                "--methods",
                method,
                "--snr-db",
            ] + [str(x) for x in snr_db] + [
                "--i-max",
                str(i_max),
                "--seed",
                str(cfg["system"]["seed"]),
            ]

            if q_table:
                eval_cmd += ["--q-table", q_table]
            if deep_ckpt:
                eval_cmd += ["--deep-checkpoint", deep_ckpt]
            # For zx methods require --z; use provided default z or 1
            z = cfg.get("parameters", {}).get("z", 1)
            eval_cmd += ["--z", str(z)]

            print("  > eval:", " ".join(eval_cmd[:8]), "...")
            rc, eval_time = run_cmd(eval_cmd)
            eval_rc = rc

            # Estimated full-run time: scale by (full_eps / smoke_eps)
            full_tabular = 208333
            full_deep = 2500
            est_full_time = 0.0
            if policy in {"tabular", "mi_tabular_zx"}:
                est_full_time = (train_time) * (full_tabular / max(1, smoke_eps_tabular))
            else:
                est_full_time = (train_time) * (full_deep / max(1, smoke_eps_deep))

            row = {
                "matrix": matrix_key,
                "method": method,
                "policy": policy,
                "train_rc": train_rc,
                "train_time_sec": round(train_time, 3),
                "eval_rc": eval_rc,
                "eval_time_sec": round(eval_time, 3),
                "estimated_full_run_time_hours": round(est_full_time / 3600.0, 3),
            }
            rows.append(row)

    # Write CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
