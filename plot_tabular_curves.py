import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    evaluate_single_method_parallel
)
from RELDEC.mdp.reward import ReldecDeltaReward

def _build_suite(h_csr: sp.csr_matrix, q_table: np.ndarray):
    from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
    suite = ReldecDecoderSuite(h_csr)
    suite.set_q_table(q_table)
    return suite

def main():
    matrix_csv = "RELDEC/matrices/H_Mackay_96_48.csv"
    snrs = [-0.1, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    episodes = 100
    workers = 8
    eval_frames = 10000
    eval_errors = 300
    code_rate = 0.5
    i_max = 50
    seed = 42

    print(f"Loading matrix from {matrix_csv}")
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    reward_fn = ReldecDeltaReward()

    from RELDEC.mdp.reward import MILocalReward, MIDeltaLocalReward

    results = {
        "flooding": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
        "reldec": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
        "dyna": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
        "dyna_mi": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []},
        "dyna_midelta": {"snrs": snrs, "fer": [], "ber": [], "avg_msgs": []}
    }

    for snr_db in snrs:
        print(f"\n{'='*50}\nEvaluating SNR: {snr_db} dB\n{'='*50}")
        
        rng = np.random.default_rng(seed)
        snr_schedule_db = build_training_snr_schedule([snr_db], episodes, rng)

        # Train RELDEC
        reldec_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), reward_fn)
        reldec_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})
        
        # Train Dyna (RELDEC Delta)
        dyna_trainer = DynaTrainer(h_csr, DynaHyperParams(), reward_fn)
        dyna_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})

        # Train Dyna (MI)
        dyna_mi_trainer = DynaTrainer(h_csr, DynaHyperParams(), MILocalReward())
        dyna_mi_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})

        # Train Dyna (MI Delta)
        dyna_midelta_trainer = DynaTrainer(h_csr, DynaHyperParams(), MIDeltaLocalReward())
        dyna_midelta_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})

        # Evaluate Flooding
        print(f"Running Flooding...")
        suite = _build_suite(h_csr, np.zeros((64, h_csr.shape[0])))
        stats = evaluate_single_method_parallel(
            suite=suite, method="flooding", snr_db=snr_db, code_rate=code_rate, i_max=i_max,
            target_frame_errors=eval_errors, max_frames=eval_frames, rng=np.random.default_rng(seed + 100), n_workers=workers
        )
        row = stats.summary(snr_db)
        results["flooding"]["fer"].append(row["fer"])
        results["flooding"]["ber"].append(row["ber"])
        results["flooding"]["avg_msgs"].append(row["avg_messages"])

        # Evaluate RELDEC
        print(f"Running RELDEC...")
        suite = _build_suite(h_csr, reldec_trainer.q_table)
        stats = evaluate_single_method_parallel(
            suite=suite, method="reldec", snr_db=snr_db, code_rate=code_rate, i_max=i_max,
            target_frame_errors=eval_errors, max_frames=eval_frames, rng=np.random.default_rng(seed + 200), n_workers=workers
        )
        row = stats.summary(snr_db)
        results["reldec"]["fer"].append(row["fer"])
        results["reldec"]["ber"].append(row["ber"])
        results["reldec"]["avg_msgs"].append(row["avg_messages"])

        # Evaluate Dyna
        print(f"Running Dyna (RELDEC Delta)...")
        suite = _build_suite(h_csr, dyna_trainer.q_table)
        stats = evaluate_single_method_parallel(
            suite=suite, method="reldec", snr_db=snr_db, code_rate=code_rate, i_max=i_max,
            target_frame_errors=eval_errors, max_frames=eval_frames, rng=np.random.default_rng(seed + 300), n_workers=workers
        )
        row = stats.summary(snr_db)
        results["dyna"]["fer"].append(row["fer"])
        results["dyna"]["ber"].append(row["ber"])
        results["dyna"]["avg_msgs"].append(row["avg_messages"])

        # Evaluate Dyna MI
        print(f"Running Dyna (MI)...")
        suite = _build_suite(h_csr, dyna_mi_trainer.q_table)
        stats = evaluate_single_method_parallel(
            suite=suite, method="reldec", snr_db=snr_db, code_rate=code_rate, i_max=i_max,
            target_frame_errors=eval_errors, max_frames=eval_frames, rng=np.random.default_rng(seed + 400), n_workers=workers
        )
        row = stats.summary(snr_db)
        results["dyna_mi"]["fer"].append(row["fer"])
        results["dyna_mi"]["ber"].append(row["ber"])
        results["dyna_mi"]["avg_msgs"].append(row["avg_messages"])

        # Evaluate Dyna MI Delta
        print(f"Running Dyna (MI Delta)...")
        suite = _build_suite(h_csr, dyna_midelta_trainer.q_table)
        stats = evaluate_single_method_parallel(
            suite=suite, method="reldec", snr_db=snr_db, code_rate=code_rate, i_max=i_max,
            target_frame_errors=eval_errors, max_frames=eval_frames, rng=np.random.default_rng(seed + 500), n_workers=workers
        )
        row = stats.summary(snr_db)
        results["dyna_midelta"]["fer"].append(row["fer"])
        results["dyna_midelta"]["ber"].append(row["ber"])
        results["dyna_midelta"]["avg_msgs"].append(row["avg_messages"])

    # --- Plotting ---
    plt.figure(figsize=(15, 5))

    # FER Plot
    plt.subplot(1, 3, 1)
    for method, data in results.items():
        plt.semilogy(data["snrs"], data["fer"], marker='o', label=method.upper())
    plt.title("Frame Error Rate (FER)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("FER")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # BER Plot
    plt.subplot(1, 3, 2)
    for method, data in results.items():
        plt.semilogy(data["snrs"], data["ber"], marker='o', label=method.upper())
    plt.title("Bit Error Rate (BER)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Avg Messages Plot
    plt.subplot(1, 3, 3)
    for method, data in results.items():
        plt.plot(data["snrs"], data["avg_msgs"], marker='o', label=method.upper())
    plt.title("Average Messages Passed")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Messages")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    out_path = "/root/.gemini/antigravity-ide/brain/1530c361-ee94-43f5-b810-de865301a536/tabular_vs_dyna_plots.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plots saved to {out_path}")

if __name__ == "__main__":
    main()
