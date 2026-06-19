#!/usr/bin/env python3
"""
Evaluate DeepDynaDiscrete (Dyna-Q with MI-vector state) vs Flooding vs RELDEC vs Tabular Dyna.
State: MI vector (48-dim in [0,1]), Action: single cluster scheduling.
"""
import time
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor, as_completed

from RELDEC.algorithms.reldec_core import (
    load_parity_check_from_sparse_csv,
    build_training_snr_schedule,
    ReldecHyperParams,
    ReldecTrainer,
    DynaHyperParams,
    DynaTrainer,
    evaluate_single_method_parallel,
    merge_method_stats,
    MethodStats,
)
from RELDEC.algorithms.reldec_deep import (
    DeepDynaConfig,
    DeepDynaDiscreteTrainer,
    DeepDynaDiscreteDecoder,
    evaluate_deep_dyna_discrete_method,
    QNetwork,
)
from RELDEC.mdp.reward import ReldecDeltaReward


# ── helpers ─────────────────────────────────────────────────────────────────

def _build_reldec_suite(h_csr, q_table):
    from RELDEC.algorithms.reldec_core import ReldecDecoderSuite
    suite = ReldecDecoderSuite(h_csr)
    suite.set_q_table(q_table)
    return suite


def _deep_dyna_discrete_worker(args):
    """Worker function for parallel evaluation of DeepDynaDiscrete."""
    import scipy.sparse as sp
    from RELDEC.algorithms.reldec_deep import DeepDynaDiscreteDecoder, QNetwork, evaluate_deep_dyna_discrete_method
    import numpy as np

    (h_data, h_indices, h_indptr, h_shape,
     state_dict, num_actions, hidden_dim, cluster_size,
     snr_db, code_rate, i_max,
     n_frames, target_errors, seed) = args

    h_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape, dtype=np.uint8)
    state_dim = num_actions  # MI state: one float per cluster

    net = QNetwork(state_dim, num_actions, hidden_dim)
    net.load_state_dict(state_dict)

    decoder = DeepDynaDiscreteDecoder(h_csr, net, cluster_size=cluster_size, device="cpu")
    rng = np.random.default_rng(seed)

    return evaluate_deep_dyna_discrete_method(
        decoder=decoder,
        snr_db=snr_db,
        code_rate=code_rate,
        i_max=i_max,
        target_frame_errors=target_errors,
        max_frames=n_frames,
        rng=rng,
        all_zero_only=True,
    )


def eval_deep_dyna_discrete_parallel(
    h_csr, trainer: DeepDynaDiscreteTrainer, snr_db, code_rate, i_max,
    eval_frames, eval_errors, workers, seed,
):
    state_dict_cpu = {k: v.cpu() for k, v in trainer.online_net.state_dict().items()}
    h_args = (h_csr.data, h_csr.indices, h_csr.indptr, h_csr.shape)
    cfg = trainer.config

    fpw = max(1, eval_frames // workers)
    fpl = eval_frames - fpw * (workers - 1)
    epw = max(1, eval_errors // workers)
    epl = eval_errors - epw * (workers - 1)
    worker_frames = [fpw] * (workers - 1) + [fpl]
    worker_errors = [epw] * (workers - 1) + [epl]
    worker_seeds  = [seed + 200 + i for i in range(workers)]

    worker_args = [
        (*h_args, state_dict_cpu, trainer.num_actions, cfg.hidden_dim, cfg.cluster_size,
         snr_db, code_rate, i_max, worker_frames[i], worker_errors[i], worker_seeds[i])
        for i in range(workers)
    ]

    partials = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_deep_dyna_discrete_worker, a): i for i, a in enumerate(worker_args)}
        for fut in as_completed(futs):
            partials.append(fut.result())

    return merge_method_stats(partials)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv",  default="RELDEC/matrices/H_Mackay_96_48.csv")
    parser.add_argument("--snr-db",      type=float, default=2.0)
    parser.add_argument("--episodes",    type=int,   default=100)
    parser.add_argument("--workers",     type=int,   default=8)
    parser.add_argument("--eval-frames", type=int,   default=10000)
    parser.add_argument("--eval-errors", type=int,   default=300)
    args = parser.parse_args()

    snr_db     = args.snr_db
    episodes   = args.episodes
    workers    = args.workers
    code_rate  = 0.5
    i_max      = 50
    seed       = 42

    print(f"Loading matrix from {args.matrix_csv}")
    h = load_parity_check_from_sparse_csv(args.matrix_csv)
    reward_fn = ReldecDeltaReward()
    rng = np.random.default_rng(seed)
    snr_schedule = build_training_snr_schedule([snr_db], episodes, rng)
    run_config   = {"snr_schedule_db": snr_schedule, "code_rate": code_rate, "seed": seed}

    results = {}  # method -> (stats, eval_time, train_time)

    # ── 1. Train RELDEC ────────────────────────────────────────────────────
    print(f"\n--- Training Tabular RELDEC ({episodes} eps) ---")
    r_trainer = ReldecTrainer(h, ReldecHyperParams(), reward_fn)
    t0 = time.time()
    r_trainer.train(run_config)
    reldec_train_time = time.time() - t0
    print(f"  done in {reldec_train_time:.2f}s")

    # ── 2. Train Tabular Dyna ──────────────────────────────────────────────
    print(f"\n--- Training Tabular Dyna ({episodes} eps) ---")
    d_trainer = DynaTrainer(h, DynaHyperParams(), reward_fn)
    t0 = time.time()
    d_trainer.train(run_config)
    dyna_train_time = time.time() - t0
    print(f"  done in {dyna_train_time:.2f}s")

    # ── 3. Train DeepDynaDiscrete ────────────────────────────────────────────────
    print(f"\n--- Training DeepDynaDiscrete ({episodes} eps) ---")
    mi_config = DeepDynaConfig(
        policy_label="deep_dyna_discrete",
        cluster_size=1,
        n_planning_steps=10,
        hidden_dim=128,
        learning_rate=1e-3,
        replay_capacity=5000,
        replay_warmup=50,
        batch_size=32,
        target_sync_steps=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=episodes * i_max,
        gamma=0.9,
    )
    mi_trainer = DeepDynaDiscreteTrainer(h, mi_config, beta_discount=0.9, l_max=i_max, device="cpu")
    t0 = time.time()
    mi_prog = mi_trainer.train(run_config)
    mi_train_time = time.time() - t0
    print(f"  done in {mi_train_time:.2f}s  |  total_reward={mi_prog.reward_sum:.1f}")

    # ── 4. Evaluate all methods ────────────────────────────────────────────
    print(f"\n--- Evaluating (Workers: {workers}) ---")

    # Flooding
    t0 = time.time()
    flood_suite = _build_reldec_suite(h, np.zeros((64, h.shape[0])))
    flood_stats = evaluate_single_method_parallel(
        suite=flood_suite, method="flooding",
        snr_db=snr_db, code_rate=code_rate, i_max=i_max,
        target_frame_errors=args.eval_errors, max_frames=args.eval_frames,
        rng=np.random.default_rng(seed+10), n_workers=workers,
    )
    flood_stats.method = "flooding"
    results["flooding"] = (flood_stats, time.time()-t0, 0.0)

    # RELDEC
    t0 = time.time()
    rel_suite = _build_reldec_suite(h, r_trainer.q_table)
    rel_stats = evaluate_single_method_parallel(
        suite=rel_suite, method="reldec",
        snr_db=snr_db, code_rate=code_rate, i_max=i_max,
        target_frame_errors=args.eval_errors, max_frames=args.eval_frames,
        rng=np.random.default_rng(seed+20), n_workers=workers,
    )
    rel_stats.method = "reldec"
    results["reldec"] = (rel_stats, time.time()-t0, reldec_train_time)

    # Tabular Dyna (uses reldec decoder with its q_table)
    t0 = time.time()
    dyn_suite = _build_reldec_suite(h, d_trainer.q_table)
    dyn_stats = evaluate_single_method_parallel(
        suite=dyn_suite, method="reldec",
        snr_db=snr_db, code_rate=code_rate, i_max=i_max,
        target_frame_errors=args.eval_errors, max_frames=args.eval_frames,
        rng=np.random.default_rng(seed+30), n_workers=workers,
    )
    dyn_stats.method = "dyna"
    results["dyna"] = (dyn_stats, time.time()-t0, dyna_train_time)

    # DeepDynaDiscrete
    t0 = time.time()
    mi_stats = eval_deep_dyna_discrete_parallel(
        h, mi_trainer, snr_db, code_rate, i_max,
        args.eval_frames, args.eval_errors, workers, seed,
    )
    results["deep_dyna_discrete"] = (mi_stats, time.time()-t0, mi_train_time)

    # ── 5. Print table ─────────────────────────────────────────────────────
    print(f"\n{'Method':15s} {'FER':>12s} {'BER':>12s} {'AvgMsgs':>10s} {'EvalTime':>10s} {'TrainTime':>10s}")
    print("-" * 75)
    for name, (stats, etime, ttime) in results.items():
        row = stats.summary(snr_db=snr_db)
        ms_per_frame = (etime / row["frames"] * 1000) if row["frames"] else 0
        print(
            f"{name:15s} {row['fer']:>12.6e} {row['ber']:>12.6e} "
            f"{row['avg_messages']:>10.2f} {etime:>8.2f}s  {ttime:>8.2f}s"
        )


if __name__ == "__main__":
    main()
