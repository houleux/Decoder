#!/usr/bin/env python3
"""Run the Mackay benchmark across all selected methods.

This script mirrors the partial/full launchers but is scoped to the Mackay
matrix only, uses the benchmark config that targets 2,500 episodes per SNR,
and writes per-method logs plus a timing CSV for tracking.
"""

from __future__ import annotations

import csv
import datetime
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

print(f"[startup] Python: {sys.executable} {sys.version.split()[0]}", flush=True)

try:
    import yaml
except ImportError:
    print("ERROR: 'pyyaml' not installed. Run: pip install pyyaml", flush=True)
    sys.exit(1)

try:
    import torch
    _CUDA_OK = torch.cuda.is_available()
    _NUM_GPUS = torch.cuda.device_count() if _CUDA_OK else 0
    print(f"[startup] torch={torch.__version__} cuda_available={_CUDA_OK} num_gpus={_NUM_GPUS}", flush=True)
except ImportError:
    _CUDA_OK = False
    _NUM_GPUS = 0
    print("[startup] torch not found — deep methods will fail", flush=True)
except Exception as e:
    _CUDA_OK = False
    _NUM_GPUS = 0
    print(f"[startup] torch import warning: {e}", flush=True)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRAINING_METHODS = {
    "reldec",
    "deep_reldec_zx",
    "mi_dqn_zx",
    "mi_tabular_zx",
    "augmented_max_avg_zx",
    "augmented_max_zx",
    "augmented_average_zx",
    "tabular_augmented_max_avg_zx",
    "tabular_augmented_max_zx",
    "tabular_augmented_average_zx",
    "reldec_misq_global",
    "reldec_misq_local",
    "rel_delta",
}

# All tabular-family policies (use tabular_eps, not deep_eps)
TABULAR_POLICIES = {
    "tabular",
    "mi_tabular_zx",
    "reldec_misq_global",
    "reldec_misq_local",
    "rel_delta",
    "tabular_augmented_max_avg_zx",
    "tabular_augmented_max_zx",
    "tabular_augmented_average_zx",
}


def policy_for_method(method: str) -> str:
    if method == "reldec":
        return "tabular"
    if method in ("reldec_misq_global", "reldec_misq_local", "rel_delta"):
        return method
    if method.startswith("tabular_augmented"):
        return method
    if method.startswith("mi_tabular"):
        return "mi_tabular_zx"
    if method.endswith("_zx") or method.startswith("deep_") or method.startswith("mi_dqn") or method.startswith("augmented_"):
        return "deep_zx"
    return "tabular"


def build_env(cuda_visible: str | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    project_root = str(ROOT)
    ldpc_path = str(ROOT.parent / "ldpc" / "src_python")
    existing = env.get("PYTHONPATH", "")
    paths = [project_root, ldpc_path]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = ":".join(paths)
    # Only pin to a specific GPU if it's within the actual allocated count.
    # Slurm may have already set CUDA_VISIBLE_DEVICES; don't override with an
    # out-of-range index (e.g. config has gpu_workers=4 but job has 2 GPUs).
    if cuda_visible is not None and _NUM_GPUS > 0 and int(cuda_visible) < _NUM_GPUS:
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
    method = task["method"]
    matrix_key = task["matrix_key"]
    matrix_csv = task["matrix_csv"]
    policy = task["policy"]
    tabular_eps = task["tabular_eps"]
    deep_eps = task["deep_eps"]
    snr_db = task["snr_db"]
    i_max = task["i_max"]
    target_frame_errors = task.get("target_frame_errors", 300)
    max_frames = task.get("max_frames", 200000)
    z_value = task.get("z", 1)
    run_tag = task["run_tag"]

    env = build_env(cuda_visible=str(gpu_id) if gpu_id is not None else None)

    # Isolate checkpoints and logs per-z and per-run so different z values
    # don't share cached results and each run gets fresh names.
    z_tag = f"z{z_value}"
    ckpt_dir = ROOT / "checkpoints" / f"{run_tag}_{method}_{matrix_key}_{z_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_base_dir = ROOT / "logs" / f"wran_bench_{run_tag}"
    log_dir = log_base_dir / z_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = log_dir / f"{method}_{matrix_key}.train.log"
    eval_log_path = log_dir / f"{method}_{matrix_key}.eval.log"

    result = {
        "z": z_value,
        "matrix": matrix_key,
        "method": method,
        "policy": policy,
        "train_time_sec": None,
        "eval_time_sec": None,
        "train_rc": None,
        "eval_rc": None,
        "gpu_id": gpu_id,
    }

    eps = tabular_eps if policy in TABULAR_POLICIES else deep_eps

    if method in TRAINING_METHODS:
        train_cmd = [
            sys.executable,
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
            "cuda" if (gpu_id is not None and _CUDA_OK) else "cpu",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
        if policy == "mi_tabular_zx":
            train_cmd += ["--mi-bins", str(task.get("mi_bins", 21))]
        if policy != "tabular":
            train_cmd += ["--z", str(z_value)]

        rc, elapsed = run_process(train_cmd, env, train_log_path)
        result["train_time_sec"] = round(elapsed, 3)
        result["train_rc"] = rc
        if rc != 0:
            result["eval_time_sec"] = 0.0
            result["eval_rc"] = 1
            return result
    else:
        result["train_time_sec"] = 0.0
        result["train_rc"] = 0

    q_table_path = ckpt_dir / "q_table_final.npy"
    deep_ckpt_path = ckpt_dir / "dqn_final.npz"

    # Determine if this is a tabular-augmented method (needs separate flag)
    is_tabular_augmented = method.startswith("tabular_augmented")

    eval_cmd = [
        sys.executable,
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
        "--target-frame-errors",
        str(target_frame_errors),
        "--max-frames",
        str(max_frames),
        "--seed",
        str(task.get("seed", 42)),
    ]
    if q_table_path.exists():
        if is_tabular_augmented:
            eval_cmd += ["--tabular-augmented-q-table", str(q_table_path)]
        else:
            eval_cmd += ["--q-table", str(q_table_path)]
    if deep_ckpt_path.exists():
        eval_cmd += ["--deep-checkpoint", str(deep_ckpt_path)]
    eval_cmd += ["--z", str(z_value)]

    rc, elapsed = run_process(eval_cmd, env, eval_log_path)
    result["eval_time_sec"] = round(elapsed, 3)
    result["eval_rc"] = rc
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run full tabular WRAN benchmark.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--z", type=int, default=None, help="Override z value from config")
    parser.add_argument("--run-tag", type=str, default=None, help="Reuse an existing run_tag to resume interrupted jobs (e.g. 0613_013444)")
    args = parser.parse_args()

    config_path = args.config
    print(f"[main] ROOT={ROOT}", flush=True)
    print(f"[main] config={config_path}", flush=True)

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", flush=True)
        return 1

    cfg = yaml.safe_load(config_path.read_text())
    benchmarking = cfg["benchmarking"]
    methods = benchmarking["methods"]
    tabular_eps = benchmarking["train"]["tabular"]["episodes_per_snr"]
    deep_eps = benchmarking["train"]["deep"]["episodes_per_snr"]
    snr_db = benchmarking["evaluation"]["snr_db"]
    i_max = benchmarking["evaluation"].get("i_max", 50)
    target_frame_errors = benchmarking["evaluation"].get("target_frame_errors", 300)
    max_frames = benchmarking["evaluation"].get("max_frames", 200000)
    system = cfg.get("system", {})
    gpu_workers = int(system.get("gpu_workers", 4))
    cpu_workers = int(system.get("cpu_workers", 40))
    z_value = args.z if args.z is not None else int(cfg.get("parameters", {}).get("z", 6))
    mi_bins = int(cfg.get("parameters", {}).get("mi_bins", 21))
    flooding_only_z1 = bool(benchmarking.get("flooding_only_z1", False))

    # Use provided run_tag to resume, else generate a fresh one
    run_tag = args.run_tag if args.run_tag else datetime.datetime.now().strftime("%m%d_%H%M%S")
    print(f"[main] run_tag={run_tag} z={z_value}", flush=True)

    matrix_csv = ROOT / "matrices" / "WRAN_irreg_384_256.csv"
    if not matrix_csv.exists():
        print(f"ERROR: Missing WRAN matrix: {matrix_csv}", flush=True)
        return 1

    # Filter methods: skip flooding for z != 1 if flooding_only_z1 is set
    active_methods = []
    for method in methods:
        if flooding_only_z1 and method == "flooding" and z_value != 1:
            print(f"[main] Skipping flooding for z={z_value} (flooding_only_z1=True)", flush=True)
            continue
        active_methods.append(method)

    tasks = [
        {
            "matrix_key": "wran",
            "matrix_csv": matrix_csv,
            "method": method,
            "policy": policy_for_method(method),
            "tabular_eps": tabular_eps,
            "deep_eps": deep_eps,
            "snr_db": snr_db,
            "i_max": i_max,
            "target_frame_errors": target_frame_errors,
            "max_frames": max_frames,
            "seed": int(system.get("seed", 42)),
            "z": z_value,
            "mi_bins": mi_bins,
            "run_tag": run_tag,
        }
        for method in active_methods
    ]

    cpu_tasks = [task for task in tasks if not (task["policy"].startswith("deep") or task["policy"].startswith("mi_dqn") or task["policy"].startswith("augmented"))]
    gpu_tasks = [task for task in tasks if task not in cpu_tasks]
    results = []

    print(
        f"Launching {len(tasks)} WRAN tasks with {len(cpu_tasks)} CPU jobs and {len(gpu_tasks)} GPU jobs "
        f"(CPUs={cpu_workers}, GPUs={gpu_workers})"
    )

    cpu_futures = []
    gpu_futures = []

    # Cap gpu_workers to however many GPUs are actually available in this job.
    effective_gpu_workers = min(gpu_workers, max(1, _NUM_GPUS)) if _CUDA_OK else 1
    if effective_gpu_workers != gpu_workers:
        print(f"[main] gpu_workers={gpu_workers} capped to {effective_gpu_workers} (actual GPUs: {_NUM_GPUS})", flush=True)

    with ThreadPoolExecutor(max_workers=min(cpu_workers, max(1, len(cpu_tasks)))) as cpu_ex:
        with ThreadPoolExecutor(max_workers=min(effective_gpu_workers, max(1, len(gpu_tasks)))) as gpu_ex:
            gpu_idx = 0
            for task in cpu_tasks:
                cpu_futures.append(cpu_ex.submit(run_task, task, None))
            for task in gpu_tasks:
                # Cycle only through real GPU indices [0 .. _NUM_GPUS-1]
                assigned_gpu = gpu_idx % max(1, _NUM_GPUS) if _CUDA_OK else None
                gpu_idx += 1
                gpu_futures.append(gpu_ex.submit(run_task, task, assigned_gpu))

            for fut in as_completed(cpu_futures + gpu_futures):
                r = fut.result()
                results.append(r)
                print(f"Completed: z={r['z']} {r['matrix']}/{r['method']} train={r['train_time_sec']}s eval={r['eval_time_sec']}s rc_train={r['train_rc']} rc_eval={r['eval_rc']} gpu={r['gpu_id']}")

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_csv = RESULTS_DIR / f"wran_benchmark_timings_z{z_value}.csv"
        with open(results_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote results to {results_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())