#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from RELDEC.algorithms.reldec_core import (
    get_code_preset,
    load_parity_check_from_sparse_csv,
    ReldecHyperParams,
    evaluate_single_method,
    ReldecDecoderSuite,
    TrainingConfig,
)
from RELDEC.trainer_factory import TrainerFactory
from RELDEC.mdp.reward import MeanNeighborSignReward


def main() -> None:
    preset = get_code_preset("mackay")
    matrix_csv = str(preset.matrix_csv)
    print(f"Loading matrix: {matrix_csv}")
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)

    hyper = ReldecHyperParams(alpha=0.1, beta=0.9, epsilon=0.6, l_max=5)

    policies = ["tabular", "reldec_misq_local", "reldec_misq_global", "rel_delta"]
    methods = ["reldec", "reldec_misq_local", "reldec_misq_global", "rel_delta"]
    eval_methods = ["flooding"] + methods

    q_tables = {}

    # Quick train
    train_config = {
        "snr_schedule_db": np.array([2.0, 2.5] * 20, dtype=np.float64),
        "code_rate": 0.5,
        "seed": 42,
    }

    for pol in policies:
        print(f"Training {pol}...")
        config = TrainingConfig(
            code="mackay",
            matrix_csv=matrix_csv,
            train_snr_db=[2.0, 2.5],
            episodes_per_snr=20,
            code_rate=0.5,
            seed=42,
            hyperparams=hyper,
        )
        trainer = TrainerFactory.create_tabular_trainer(h_csr, config, pol)
        progress = trainer.train(train_config)
        q_tables[pol] = trainer.q_table
        print(f"  Mean reward: {progress.mean_reward():.6f}")

    snr_points = [0.5, 1.0, 1.5, 2.0, 2.5]
    rng = np.random.default_rng(12345)

    results = {m: [] for m in eval_methods}

    for method in eval_methods:
        print(f"Evaluating {method}...")
        # Note: reward_fn doesn't matter for evaluation, so we don't pass it.
        # ReldecDecoderSuite only requires h_csr and optionally q_table.
        suite = ReldecDecoderSuite(h_csr)
        
        if method != "flooding":
            pol = "tabular" if method == "reldec" else method
            suite.set_q_table(q_tables[pol])

        for snr in snr_points:
            stats = evaluate_single_method(
                suite=suite,
                method=method,
                snr_db=snr,
                code_rate=0.5,
                i_max=20,
                target_frame_errors=10,
                max_frames=50,
                rng=rng,
                all_zero_only=True,
            )
            results[method].append(stats)
            summary = stats.summary(snr)
            print(f"  SNR {snr:.1f}: FER={summary['fer']:.4f}, BER={summary['ber']:.4f}")

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Tabular Variants vs Flooding (MacKay, l_max=5)", fontsize=14, fontweight="bold")

    for m in eval_methods:
        fer = [r.summary(snr)["fer"] for r, snr in zip(results[m], snr_points)]
        ber = [r.summary(snr)["ber"] for r, snr in zip(results[m], snr_points)]
        msgs = [r.summary(snr)["avg_messages"] for r, snr in zip(results[m], snr_points)]

        axs[0].plot(snr_points, fer, marker="o", label=m)
        axs[1].plot(snr_points, ber, marker="o", label=m)
        axs[2].plot(snr_points, msgs, marker="o", label=m)

    for ax in axs[:2]:
        ax.set_yscale("log")
    for ax in axs:
        ax.set_xlabel("SNR (dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axs[0].set_title("FER")
    axs[1].set_title("BER")
    axs[2].set_title("Avg Messages")

    plt.tight_layout()
    out_path = Path("smoke_variants_plot.png").resolve()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
