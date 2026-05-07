from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reldec_deep import (
    DeepDqnConfig,
    DeepReldecTrainer,
    DeepTrainingCheckpoint,
    MiTabularQTrainer,
    load_deep_training_checkpoint,
    save_deep_training_checkpoint,
)
from reldec_core import (
    THIS_DIR,
    ReldecHyperParams,
    ReldecTrainer,
    TrainProgress,
    TrainingCheckpoint,
    TrainingConfig,
    build_training_snr_schedule,
    get_code_preset,
    load_parity_check_from_sparse_csv,
    load_training_checkpoint,
    nominal_code_rate,
    save_training_checkpoint,
    train_reldec,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RELDEC (tabular z=1, Deep RELDEC DQN z=1/z=2, MI-DQN z=2, or MI-tabular z=2)."
    )
    parser.add_argument("--code", choices=["ab", "wran"], default="ab")
    parser.add_argument("--matrix-csv", type=str, default=None)
    parser.add_argument("--snr-db", type=float, nargs="+", default=None)
    parser.add_argument("--episodes-per-snr", type=int, default=2500)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--l-max", type=int, default=50)
    parser.add_argument("--code-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every-episodes", type=int, default=250)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--policy-type",
        choices=["tabular", "mi_tabular_z2", "deep_z1", "deep_z2", "mi_dqn_z2"],
        default="tabular",
    )
    parser.add_argument("--mi-bins", type=int, default=21)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dqn-hidden-dim", type=int, default=128)
    parser.add_argument("--dqn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dqn-replay-capacity", type=int, default=20000)
    parser.add_argument("--dqn-replay-warmup", type=int, default=1000)
    parser.add_argument("--dqn-batch-size", type=int, default=128)
    parser.add_argument("--dqn-target-sync-steps", type=int, default=200)
    parser.add_argument("--dqn-train-every-steps", type=int, default=1)
    parser.add_argument("--dqn-epsilon-start", type=float, default=0.6)
    parser.add_argument("--dqn-epsilon-end", type=float, default=0.05)
    parser.add_argument("--dqn-epsilon-decay-steps", type=int, default=10000)
    return parser.parse_args()


def _build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    preset = get_code_preset(args.code)
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else preset.matrix_csv
    train_snr_db = tuple(args.snr_db) if args.snr_db else preset.train_snr_db

    h = load_parity_check_from_sparse_csv(matrix_csv)
    code_rate = args.code_rate if args.code_rate is not None else nominal_code_rate(h)

    hyperparams = ReldecHyperParams(
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        l_max=args.l_max,
    )

    return TrainingConfig(
        code=args.code,
        matrix_csv=str(matrix_csv),
        train_snr_db=train_snr_db,
        episodes_per_snr=args.episodes_per_snr,
        code_rate=code_rate,
        seed=args.seed,
        hyperparams=hyperparams,
        cluster_size=1,
    )


def _cluster_size_for_policy(policy_type: str) -> int:
    if policy_type == "tabular":
        return 1
    if policy_type == "mi_tabular_z2":
        return 2
    if policy_type == "deep_z1":
        return 1
    if policy_type == "deep_z2":
        return 2
    if policy_type == "mi_dqn_z2":
        return 2
    raise ValueError(f"Unsupported policy type: {policy_type}")


def _build_deep_config(args: argparse.Namespace, cluster_size: int) -> DeepDqnConfig:
    return DeepDqnConfig(
        policy_label=str(args.policy_type),
        cluster_size=int(cluster_size),
        hidden_dim=int(args.dqn_hidden_dim),
        learning_rate=float(args.dqn_learning_rate),
        replay_capacity=int(args.dqn_replay_capacity),
        replay_warmup=int(args.dqn_replay_warmup),
        batch_size=int(args.dqn_batch_size),
        target_sync_steps=int(args.dqn_target_sync_steps),
        train_every_steps=int(args.dqn_train_every_steps),
        epsilon_start=float(args.dqn_epsilon_start),
        epsilon_end=float(args.dqn_epsilon_end),
        epsilon_decay_steps=int(args.dqn_epsilon_decay_steps),
        gamma=float(args.beta),
    )


def _validate_checkpoint_config(config: TrainingConfig, args: argparse.Namespace) -> None:
    if args.code != config.code:
        raise ValueError(f"Resume checkpoint code={config.code} but --code={args.code}")
    if args.matrix_csv is not None and str(Path(args.matrix_csv)) != config.matrix_csv:
        raise ValueError("--matrix-csv does not match resumed checkpoint config")


def _main() -> None:
    args = _parse_args()
    cluster_size = _cluster_size_for_policy(args.policy_type)

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else THIS_DIR / "checkpoints" / args.code.lower()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "checkpoint_latest.npz"
    final_path = checkpoint_dir / "checkpoint_final.npz"
    q_table_path = checkpoint_dir / "q_table_final.npy"
    dqn_final_path = checkpoint_dir / "dqn_final.npz"
    summary_path = checkpoint_dir / "training_summary.json"

    if args.resume and args.policy_type in {"tabular", "mi_tabular_z2"}:
        resume_path = Path(args.resume)
        checkpoint = load_training_checkpoint(resume_path)
        _validate_checkpoint_config(checkpoint.config, args)

        config = checkpoint.config
        progress = checkpoint.progress
        snr_schedule_db = checkpoint.snr_schedule_db

        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        if args.policy_type == "tabular":
            trainer = ReldecTrainer(h, config.hyperparams, q_table=checkpoint.q_table)
        else:
            trainer = MiTabularQTrainer(
                h_csr=h,
                alpha=config.hyperparams.alpha,
                beta=config.hyperparams.beta,
                epsilon=config.hyperparams.epsilon,
                l_max=config.hyperparams.l_max,
                cluster_size=2,
                mi_bins=int(args.mi_bins),
                q_table=checkpoint.q_table,
            )

        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint.rng_state

        print(f"[resume] loaded checkpoint: {resume_path}")
        print(f"[resume] episodes_completed={progress.episodes_completed}")
    elif args.policy_type in {"tabular", "mi_tabular_z2"}:
        config = _build_config_from_args(args)
        config = TrainingConfig(
            code=config.code,
            matrix_csv=config.matrix_csv,
            train_snr_db=config.train_snr_db,
            episodes_per_snr=config.episodes_per_snr,
            code_rate=config.code_rate,
            seed=config.seed,
            hyperparams=config.hyperparams,
            cluster_size=cluster_size,
        )
        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        if args.policy_type == "tabular":
            trainer = ReldecTrainer(h, config.hyperparams)
        else:
            trainer = MiTabularQTrainer(
                h_csr=h,
                alpha=config.hyperparams.alpha,
                beta=config.hyperparams.beta,
                epsilon=config.hyperparams.epsilon,
                l_max=config.hyperparams.l_max,
                cluster_size=2,
                mi_bins=int(args.mi_bins),
            )
        progress = TrainProgress()

        rng = np.random.default_rng(config.seed)
        rng_schedule = np.random.default_rng(config.seed + 1)
        snr_schedule_db = build_training_snr_schedule(
            config.train_snr_db,
            config.episodes_per_snr,
            rng_schedule,
        )
    elif args.resume:
        resume_path = Path(args.resume)
        checkpoint = load_deep_training_checkpoint(resume_path)
        _validate_checkpoint_config(checkpoint.config, args)

        config = checkpoint.config
        if int(config.cluster_size) != int(cluster_size):
            raise ValueError(
                f"Resume checkpoint cluster_size={config.cluster_size} but policy requires cluster_size={cluster_size}"
            )
        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        deep_config = checkpoint.dqn_config
        deep_trainer = DeepReldecTrainer(
            h_csr=h,
            dqn_config=deep_config,
            beta_discount=config.hyperparams.beta,
            l_max=config.hyperparams.l_max,
            device=args.device,
        )
        deep_trainer.import_checkpoint_payload(
            checkpoint.q_online_bytes,
            checkpoint.q_target_bytes,
            checkpoint.optimizer_bytes,
            checkpoint.global_step,
        )

        progress = checkpoint.progress
        snr_schedule_db = checkpoint.snr_schedule_db

        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint.rng_state

        print(f"[resume] loaded deep checkpoint: {resume_path}")
        print(f"[resume] episodes_completed={progress.episodes_completed}")
    else:
        config = _build_config_from_args(args)
        config = TrainingConfig(
            code=config.code,
            matrix_csv=config.matrix_csv,
            train_snr_db=config.train_snr_db,
            episodes_per_snr=config.episodes_per_snr,
            code_rate=config.code_rate,
            seed=config.seed,
            hyperparams=config.hyperparams,
            cluster_size=cluster_size,
        )

        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        deep_config = _build_deep_config(args, cluster_size=cluster_size)
        deep_trainer = DeepReldecTrainer(
            h_csr=h,
            dqn_config=deep_config,
            beta_discount=config.hyperparams.beta,
            l_max=config.hyperparams.l_max,
            device=args.device,
        )
        progress = TrainProgress()

        rng = np.random.default_rng(config.seed)
        rng_schedule = np.random.default_rng(config.seed + 1)
        snr_schedule_db = build_training_snr_schedule(
            config.train_snr_db,
            config.episodes_per_snr,
            rng_schedule,
        )

    if args.max_episodes is not None:
        max_episodes = min(int(args.max_episodes), int(snr_schedule_db.size))
        snr_schedule_db = snr_schedule_db[:max_episodes]

    total_episodes = int(snr_schedule_db.size)
    start_episode = int(progress.episodes_completed)

    if start_episode > total_episodes:
        raise ValueError(
            f"Checkpoint episodes_completed={start_episode} exceeds total episodes={total_episodes}"
        )

    print(f"[train] code={config.code} matrix={config.matrix_csv}")
    print(f"[train] H shape={h.shape} nnz={h.nnz} rate={config.code_rate:.6f}")
    print(
        "[train] episodes="
        f"{total_episodes} start={start_episode} "
        f"l_max={config.hyperparams.l_max} alpha={config.hyperparams.alpha} "
        f"beta={config.hyperparams.beta} epsilon={config.hyperparams.epsilon} "
        f"policy_type={args.policy_type} cluster_size={config.cluster_size}"
    )

    def tabular_checkpoint_callback(ep_done: int, prog: TrainProgress) -> None:
        if args.checkpoint_every_episodes <= 0:
            return
        if ep_done % args.checkpoint_every_episodes != 0:
            return

        payload = TrainingCheckpoint(
            q_table=trainer.q_table,
            config=config,
            progress=prog,
            rng_state=rng.bit_generator.state,
            snr_schedule_db=snr_schedule_db,
        )

        save_training_checkpoint(latest_path, payload)
        if not args.no_history:
            hist_path = checkpoint_dir / f"checkpoint_ep{ep_done:06d}.npz"
            save_training_checkpoint(hist_path, payload)
        print(f"[checkpoint] saved at episode {ep_done}")

    def deep_checkpoint_callback(ep_done: int, prog: TrainProgress) -> None:
        if args.checkpoint_every_episodes <= 0:
            return
        if ep_done % args.checkpoint_every_episodes != 0:
            return

        q_online, q_target, optimizer_state = deep_trainer.export_checkpoint_payload()
        payload = DeepTrainingCheckpoint(
            config=config,
            dqn_config=deep_trainer.dqn_config,
            progress=prog,
            rng_state=rng.bit_generator.state,
            snr_schedule_db=snr_schedule_db,
            global_step=deep_trainer.global_step,
            q_online_bytes=q_online,
            q_target_bytes=q_target,
            optimizer_bytes=optimizer_state,
        )

        save_deep_training_checkpoint(latest_path, payload)
        if not args.no_history:
            hist_path = checkpoint_dir / f"checkpoint_ep{ep_done:06d}.npz"
            save_deep_training_checkpoint(hist_path, payload)
        print(f"[checkpoint] saved deep checkpoint at episode {ep_done}")

    if args.policy_type in {"tabular", "mi_tabular_z2"}:
        progress = train_reldec(
            trainer=trainer,
            snr_schedule_db=snr_schedule_db,
            code_rate=config.code_rate,
            rng=rng,
            start_episode=start_episode,
            progress=progress,
            checkpoint_callback=tabular_checkpoint_callback,
            log_every=max(0, int(args.log_every)),
        )
    else:
        from reldec_core import all_zero_awgn_llr

        for ep_idx in range(start_episode, total_episodes):
            llr = all_zero_awgn_llr(
                n=deep_trainer.n,
                ebn0_db=float(snr_schedule_db[ep_idx]),
                code_rate=config.code_rate,
                rng=rng,
            )
            episode_reward, _ = deep_trainer.train_episode(llr, rng)

            progress.episodes_completed = ep_idx + 1
            progress.total_updates += config.hyperparams.l_max
            progress.reward_sum += episode_reward
            progress.reward_count += config.hyperparams.l_max

            deep_checkpoint_callback(ep_idx + 1, progress)
            if args.log_every > 0 and (ep_idx + 1) % int(args.log_every) == 0:
                print(
                    f"[train] episode={ep_idx + 1}/{total_episodes} "
                    f"mean_reward={progress.mean_reward():.6f} updates={progress.total_updates}"
                )

    if args.policy_type in {"tabular", "mi_tabular_z2"}:
        final_checkpoint = TrainingCheckpoint(
            q_table=trainer.q_table,
            config=config,
            progress=progress,
            rng_state=rng.bit_generator.state,
            snr_schedule_db=snr_schedule_db,
        )
        save_training_checkpoint(final_path, final_checkpoint)
        save_training_checkpoint(latest_path, final_checkpoint)
        np.save(q_table_path, trainer.q_table)
    else:
        q_online, q_target, optimizer_state = deep_trainer.export_checkpoint_payload()
        final_checkpoint = DeepTrainingCheckpoint(
            config=config,
            dqn_config=deep_trainer.dqn_config,
            progress=progress,
            rng_state=rng.bit_generator.state,
            snr_schedule_db=snr_schedule_db,
            global_step=deep_trainer.global_step,
            q_online_bytes=q_online,
            q_target_bytes=q_target,
            optimizer_bytes=optimizer_state,
        )
        save_deep_training_checkpoint(final_path, final_checkpoint)
        save_deep_training_checkpoint(latest_path, final_checkpoint)
        save_deep_training_checkpoint(dqn_final_path, final_checkpoint)

    summary = {
        "config": config.to_dict(),
        "h_shape": [int(h.shape[0]), int(h.shape[1])],
        "h_nnz": int(h.nnz),
        "policy_type": args.policy_type,
        "progress": {
            "episodes_completed": int(progress.episodes_completed),
            "total_updates": int(progress.total_updates),
            "mean_reward": float(progress.mean_reward()),
            "elapsed_sec": float(progress.elapsed_sec),
        },
        "artifacts": {
            "checkpoint_latest": str(latest_path),
            "checkpoint_final": str(final_path),
        },
    }
    if args.policy_type in {"tabular", "mi_tabular_z2"}:
        summary["artifacts"]["q_table_npy"] = str(q_table_path)
    else:
        summary["artifacts"]["dqn_checkpoint"] = str(dqn_final_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] training complete")
    print(f"[done] episodes_completed={progress.episodes_completed}")
    print(f"[done] total_updates={progress.total_updates}")
    print(f"[done] mean_reward={progress.mean_reward():.6f}")
    print(f"[done] artifacts: {checkpoint_dir}")


if __name__ == "__main__":
    _main()
