import numpy as np
import scipy.sparse as sp
from RELDEC.algorithms.reldec_core import load_parity_check_from_sparse_csv, build_training_snr_schedule, ReldecHyperParams, ReldecTrainer, DynaHyperParams, DynaTrainer, evaluate_single_method_parallel
from RELDEC.mdp.reward import ReldecDeltaReward

def _build_suite(h_csr: sp.csr_matrix, q_table: np.ndarray):
    from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
    suite = ReldecDecoderSuite(h_csr)
    suite.set_q_table(q_table)
    return suite

matrix_csv = "RELDEC/matrices/H_Mackay_96_48.csv"
h_csr = load_parity_check_from_sparse_csv(matrix_csv)
reward_fn = ReldecDeltaReward()
snr_db = 2.0
code_rate = 0.5
seed = 42
episodes = 100

rng = np.random.default_rng(seed)
snr_schedule_db = build_training_snr_schedule([snr_db], episodes, rng)

# RELDEC
r_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), reward_fn)
r_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})

# DYNA 10
d_trainer = DynaTrainer(h_csr, DynaHyperParams(n_planning_steps=10), reward_fn)
d_trainer.train({"snr_schedule_db": snr_schedule_db, "code_rate": code_rate, "seed": seed})

print("Evaluating...")
r_stats = evaluate_single_method_parallel(_build_suite(h_csr, r_trainer.q_table), "reldec", snr_db, code_rate, 50, 300, 10000, np.random.default_rng(seed+200), 8)
d_stats = evaluate_single_method_parallel(_build_suite(h_csr, d_trainer.q_table), "reldec", snr_db, code_rate, 50, 300, 10000, np.random.default_rng(seed+400), 8)

r_sum = r_stats.summary(snr_db)
d_sum = d_stats.summary(snr_db)

print("RELDEC:", r_sum["fer"], r_sum["avg_messages"])
print("DYNA10:", d_sum["fer"], d_sum["avg_messages"])
