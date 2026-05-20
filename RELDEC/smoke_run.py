"""Quick smoke run: train tabular RELDEC for 2 episodes to validate setup."""

from __future__ import annotations

import numpy as np
from RELDEC.algorithms.reldec_core import (
    get_code_preset,
    load_parity_check_from_sparse_csv,
    ReldecHyperParams,
    ReldecTrainer,
)


def main() -> None:
    preset = get_code_preset("mackay")
    matrix_csv = str(preset.matrix_csv)
    print(f"Loading matrix: {matrix_csv}")
    h = load_parity_check_from_sparse_csv(matrix_csv)

    hyper = ReldecHyperParams(alpha=0.1, beta=0.9, epsilon=0.6, l_max=10)
    trainer = ReldecTrainer(h_csr=h, hyperparams=hyper, cluster_size=1)

    run_config = {
        "snr_schedule_db": np.array([0.5, 1.0], dtype=np.float64),
        "code_rate": 0.5,
        "seed": 123,
    }

    progress = trainer.train(run_config)
    print("Smoke run complete:")
    print(f" Episodes completed: {progress.episodes_completed}")
    print(f" Mean reward: {progress.mean_reward():.6f}")


if __name__ == "__main__":
    main()
