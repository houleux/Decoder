#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
LDPC = ROOT / "ldpc" / "src_python"
RELDEC = ROOT / "RELDEC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LDPC))
sys.path.insert(0, str(RELDEC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RELDEC.algorithms.reldec_core import (
    get_code_preset,
    load_parity_check_from_sparse_csv,
    ReldecHyperParams,
    train_reldec,
    TrainProgress,
)
from RELDEC.algorithms.reldec_tabular_augmented import TabularAugmentedQTrainer
from RELDEC.evaluate_reldec import _main as evaluate_main
from unittest.mock import patch
import json

def run_smoke_test():
    preset = get_code_preset("mackay")
    matrix_csv = str(ROOT / preset.matrix_csv)
    print(f"Loading matrix: {matrix_csv}")
    h = load_parity_check_from_sparse_csv(matrix_csv)

    print("--- Training Tabular Augmented ---")
    trainer = TabularAugmentedQTrainer(
        h_csr=h,
        alpha=0.1,
        beta=0.9,
        epsilon=0.6,
        l_max=50,
        policy_label="tabular_augmented_max_avg_zx",
        cluster_size=6,
        mi_bins=21,
    )

    rng = np.random.default_rng(123)
    # Train for ~1000 frames
    snr_schedule_db = np.repeat([0.5, 1.0, 1.5, 2.0], 250)
    rng.shuffle(snr_schedule_db)

    progress = TrainProgress()
    
    # Train
    print("Training started...")
    for snr in snr_schedule_db:
        progress.episodes_completed += 1
        reward = trainer.train_episode(
            llr_channel=rng.normal(2.0 * snr, np.sqrt(2.0 * snr), size=h.shape[1]),
            rng=rng,
        )
        progress.reward_sum += reward
        progress.reward_count += 1
    
    print(f"Training completed. Mean reward: {progress.mean_reward():.4f}")

    q_table_path = str(ROOT / "scratch" / "tabular_augmented_smoke_qtable.npy")
    np.save(q_table_path, trainer.q_table)
    print(f"Saved Q-table to {q_table_path}")

    # Evaluate
    print("--- Evaluating Tabular Augmented ---")
    eval_csv_path = str(ROOT / "scratch" / "tabular_augmented_eval.csv")
    eval_json_path = str(ROOT / "scratch" / "tabular_augmented_eval.json")
    
    if os.path.exists(eval_json_path):
        os.remove(eval_json_path)

    args = [
        "evaluate_reldec.py",
        "--code", "mackay",
        "--matrix-csv", matrix_csv,
        "--methods", "flooding", "mi_naive_zx", "tabular_augmented_max_avg_zx",
        "--z", "6",
        "--snr-db", "0.5", "1.0", "1.5", "2.0", "2.5",
        "--i-max", "50",
        "--target-frame-errors", "50",
        "--max-frames", "1000",
        "--tabular-augmented-q-table", q_table_path,
        "--output-csv", eval_csv_path,
        "--output-json", eval_json_path,
    ]
    
    with patch.object(sys, "argv", args):
        evaluate_main()
    
    print("--- Plotting Results ---")
    import csv
    
    results = []
    with open(eval_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    methods = sorted(set(r["method"] for r in results))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["ber", "fer", "avg_messages"]
    titles = ["BER", "FER", "Avg Messages"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        for method in methods:
            m_rows = sorted([r for r in results if r["method"] == method], key=lambda x: float(x["snr_db"]))
            xs = [float(r["snr_db"]) for r in m_rows]
            ys = [float(r[metric]) for r in m_rows]
            ax.plot(xs, ys, marker='o', label=method)
            
        ax.set_title(title)
        ax.set_xlabel("SNR (dB)")
        if metric in ["ber", "fer"]:
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        
    plot_path = str(ROOT / "scratch" / "tabular_augmented_smoke_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_smoke_test()
