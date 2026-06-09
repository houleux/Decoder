#!/usr/bin/env python3
"""Run the full partial baseline (Mackay + WRAN) in parallel across GPUs/CPUs.

This script reads `RELDEC/configs/baseline_partial.yaml`, creates a task per
matrix × method, and runs training then evaluation for each task. Deep methods
are dispatched to available GPUs (round-robin); tabular/baseline methods run
on CPU workers. Results/timings are written to
`RELDEC/results/partial_full_timings.csv` and per-task logs are saved under
`RELDEC/logs/partial_full/`.

Note: This launches long-running processes. Monitor `nvidia-smi` and logs.
"""

from __future__ import annotations

import yaml
import time
import subprocess
from pathlib import Path
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "baseline_partial.yaml"
RESULTS_CSV = ROOT / "results" / "partial_full_timings.csv"
LOG_DIR = ROOT / "logs" / "partial_full"

TRAINING_METHODS = {
    "reldec",
    "deep_reldec_zx",
    "mi_dqn_zx",
    "mi_tabular_zx",
    "augmented_max_avg_zx",
    "augmented_max_zx",
    "augmented_average_zx",
}

EVAL_ONLY_METHODS = {
    "flooding",
    "random",
    "round_robin",
    "mi_naive_zx",
}


def policy_for_method(method: str) -> str:
    if method == "reldec":
        return "tabular"
    if method.endswith("_zx") or method.startswith("deep_") or method.startswith("mi_dqn") or method.startswith("augmented_"):
        return "deep_zx"
    if method.startswith("mi_tabular"):
        return "mi_tabular_zx"
    return "tabular"


def method_requires_z(method: str) -> bool:
    return method.endswith("_zx")


def matrix_z(matrix_key: str, matrix_settings: dict[str, Any], default_z: int = 1) -> int:
    for filename, settings in matrix_settings.items():
        if matrix_key in filename:
            return int(settings.get("z", default_z))
    return int(default_z)


def build_env(cuda_visible: str | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    project_root = str(ROOT)
    ldpc_path = str(ROOT / "ldpc" / "src_python")
    existing = env.get("PYTHONPATH", "")
    paths = [project_root, ldpc_path]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = ":".join(paths)
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    return env


def run_process(cmd: list[str], env: Dict[str, str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as fh:
        start = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
        elapsed = time.time() - start
    return rc, elapsed


def run_task(task: Dict[str, Any], gpu_id: int | None = None) -> Dict[str, Any]:
    matrix_key = task["matrix_key"]
    method = task["method"]
    matrix_csv = task["matrix_csv"]
    policy = task["policy"]
    tabular_eps = task["tabular_eps"]
    deep_eps = task["deep_eps"]
    snr_db = task["snr_db"]
    i_max = task["i_max"]

    env = build_env(cuda_visible=str(gpu_id) if gpu_id is not None else None)

    # checkpoint dir and task logs
    ckpt_dir = ROOT / "checkpoints" / f"{method}_{matrix_key}_full"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = LOG_DIR / f"{method}_{matrix_key}.train.log"
    eval_log_path = LOG_DIR / f"{method}_{matrix_key}.eval.log"

    res = {
        "matrix": matrix_key,
        "method": method,
        "policy": policy,
        "train_time_sec": None,
        "eval_time_sec": None,
        "train_rc": None,
        "eval_rc": None,
        "gpu_id": gpu_id,
    }

    # Determine episodes
    if policy in {"tabular", "mi_tabular_zx"}:
        eps = tabular_eps
    else:
        eps = deep_eps

    z_value = task.get("z", 1)

    # Run training only for methods that require it
    if method in TRAINING_METHODS:
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
            str(task.get("seed", 42)),
            "--device",
            "cuda" if gpu_id is not None else "cpu",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
        if method_requires_z(policy):
            train_cmd += ["--z", str(z_value)]

        rc, elapsed = run_process(train_cmd, env, train_log_path)
        res["train_time_sec"] = round(elapsed, 3)
        res["train_rc"] = rc
        if rc != 0:
            res["eval_time_sec"] = 0.0
            res["eval_rc"] = 1
            return res
    else:
        res["train_time_sec"] = 0.0
        res["train_rc"] = 0

    # Detect checkpoints
    q_table = None
    deep_ckpt = None
    if (ckpt_dir / "q_table_final.npy").exists():
        q_table = str(ckpt_dir / "q_table_final.npy")
    if (ckpt_dir / "dqn_final.npz").exists():
        deep_ckpt = str(ckpt_dir / "dqn_final.npz")

    # Build eval command
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
        str(task.get("seed", 42)),
    ]
    if q_table:
        eval_cmd += ["--q-table", q_table]
    if deep_ckpt:
        eval_cmd += ["--deep-checkpoint", deep_ckpt]
    z = task.get("z", 1)
    eval_cmd += ["--z", str(z)]

    rc, elapsed = run_process(eval_cmd, env, eval_log_path)
    res["eval_time_sec"] = round(elapsed, 3)
    res["eval_rc"] = rc

    return res


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    matrices = cfg["benchmarking"]["matrices"]
    methods = cfg["benchmarking"]["methods"]
    tabular_eps = cfg["benchmarking"]["train"]["tabular"]["episodes_per_snr"]
    deep_eps = cfg["benchmarking"]["train"]["deep"]["episodes_per_snr"]
    snr_db = cfg["benchmarking"]["evaluation"]["snr_db"]
    i_max = cfg["benchmarking"]["evaluation"].get("i_max", 50)
    system = cfg.get("system", {})
    gpu_workers = int(system.get("gpu_workers", 1))
    cpu_workers = int(system.get("cpu_workers", 8))
    matrix_settings = cfg.get("parameters", {}).get("matrix_settings", {})

    # Map short keys -> csv paths (existing matrices only)
    MATRIX_MAP = {
        "mackay": ROOT / "matrices" / "H_Mackay_96_48.csv",
        "wran": ROOT / "matrices" / "WRAN_irreg_384_256.csv",
    }

    tasks = []
    for matrix_key in matrices:
        matrix_csv = MATRIX_MAP.get(matrix_key)
        if matrix_csv is None or not matrix_csv.exists():
            print(f"Skipping missing matrix: {matrix_key}")
            continue
        for method in methods:
            tasks.append(
                {
                    "matrix_key": matrix_key,
                    "matrix_csv": matrix_csv,
                    "method": method,
                    "policy": policy_for_method(method),
                    "tabular_eps": tabular_eps,
                    "deep_eps": deep_eps,
                    "snr_db": snr_db,
                    "i_max": i_max,
                    "seed": cfg.get("system", {}).get("seed", 42),
                    "z": matrix_z(matrix_key, matrix_settings, cfg.get("parameters", {}).get("z", 1)),
                }
            )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Simple round-robin GPU assignment for deep tasks; use ThreadPool to run tasks
    gpu_idx = 0
    max_workers = min(len(tasks), gpu_workers + min(cpu_workers, 8))
    print(f"Launching {len(tasks)} tasks with up to {max_workers} workers (GPUs={gpu_workers}, CPUs={cpu_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for task in tasks:
            policy = task["policy"]
            if policy.startswith("deep") or policy.startswith("mi_dqn") or policy.startswith("augmented"):
                assigned_gpu = gpu_idx % gpu_workers
                gpu_idx += 1
                futures.append(ex.submit(run_task, task, assigned_gpu))
            else:
                futures.append(ex.submit(run_task, task, None))

        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                print(f"Completed: {r['matrix']}/{r['method']} train={r['train_time_sec']}s eval={r['eval_time_sec']}s gpu={r['gpu_id']}")
            except Exception as e:
                print("Task failed:", e)

    # Save CSV
    if results:
        RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"Wrote results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
