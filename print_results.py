import numpy as np
import scipy.sparse as sp

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    evaluate_single_method_parallel,
    ReldecDecoderSuite
)
from RELDEC.mdp.reward import ReldecDeltaReward

def main():
    matrix_csv = "RELDEC/matrices/H_Mackay_96_48.csv"
    h_csr = load_parity_check_from_sparse_csv(matrix_csv)
    reward_fn = ReldecDeltaReward()
    snr = 2.0
    episodes = 100
    code_rate = 0.5
    seed = 42

    rng = np.random.default_rng(seed)
    snr_schedule = build_training_snr_schedule([snr], episodes, rng)
    
    reldec_trainer = ReldecTrainer(h_csr, ReldecHyperParams(), reward_fn)
    reldec_trainer.train({"snr_schedule_db": snr_schedule, "code_rate": code_rate, "seed": seed})

    dyna_trainer = DynaTrainer(h_csr, DynaHyperParams(n_planning_steps=10), reward_fn)
    dyna_trainer.train({"snr_schedule_db": snr_schedule, "code_rate": code_rate, "seed": seed})

    # Evaluator
    eval_kwargs = dict(snr_db=snr, code_rate=code_rate, i_max=50, target_frame_errors=300, max_frames=10000, n_workers=8)
    
    suite_f = ReldecDecoderSuite(h_csr)
    stat_f = evaluate_single_method_parallel(suite=suite_f, method="flooding", rng=np.random.default_rng(100), **eval_kwargs).summary(snr)
    
    suite_r = ReldecDecoderSuite(h_csr)
    suite_r.set_q_table(reldec_trainer.q_table)
    stat_r = evaluate_single_method_parallel(suite=suite_r, method="reldec", rng=np.random.default_rng(200), **eval_kwargs).summary(snr)
    
    suite_d = ReldecDecoderSuite(h_csr)
    suite_d.set_q_table(dyna_trainer.q_table)
    stat_d = evaluate_single_method_parallel(suite=suite_d, method="reldec", rng=np.random.default_rng(300), **eval_kwargs).summary(snr)
    
    print(f"Flooding: FER={stat_f['fer']:.4f}, Msgs={stat_f['avg_messages']:.1f}")
    print(f"RELDEC  : FER={stat_r['fer']:.4f}, Msgs={stat_r['avg_messages']:.1f}")
    print(f"Dyna    : FER={stat_d['fer']:.4f}, Msgs={stat_d['avg_messages']:.1f}")

if __name__ == "__main__":
    main()
