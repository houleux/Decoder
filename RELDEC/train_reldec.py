from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
        description="Train RELDEC tabular Q-table (z=1 CN cluster scheduling)."
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


def _validate_checkpoint_config(config: TrainingConfig, args: argparse.Namespace) -> None:
    if args.code != config.code:
        raise ValueError(f"Resume checkpoint code={config.code} but --code={args.code}")
    if args.matrix_csv is not None and str(Path(args.matrix_csv)) != config.matrix_csv:
        raise ValueError("--matrix-csv does not match resumed checkpoint config")


def _main() -> None:
    args = _parse_args()

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else THIS_DIR / "checkpoints" / args.code.lower()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "checkpoint_latest.npz"
    final_path = checkpoint_dir / "checkpoint_final.npz"
    q_table_path = checkpoint_dir / "q_table_final.npy"
    summary_path = checkpoint_dir / "training_summary.json"

    if args.resume:
        resume_path = Path(args.resume)
        checkpoint = load_training_checkpoint(resume_path)
        _validate_checkpoint_config(checkpoint.config, args)

        config = checkpoint.config
        progress = checkpoint.progress
        snr_schedule_db = checkpoint.snr_schedule_db

        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        trainer = ReldecTrainer(h, config.hyperparams, q_table=checkpoint.q_table)

        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint.rng_state

        print(f"[resume] loaded checkpoint: {resume_path}")
        print(f"[resume] episodes_completed={progress.episodes_completed}")
    else:
        config = _build_config_from_args(args)
        h = load_parity_check_from_sparse_csv(config.matrix_csv)
        trainer = ReldecTrainer(h, config.hyperparams)
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
        f"beta={config.hyperparams.beta} epsilon={config.hyperparams.epsilon}"
    )

    def checkpoint_callback(ep_done: int, prog: TrainProgress) -> None:
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

    progress = train_reldec(
        trainer=trainer,
        snr_schedule_db=snr_schedule_db,
        code_rate=config.code_rate,
        rng=rng,
        start_episode=start_episode,
        progress=progress,
        checkpoint_callback=checkpoint_callback,
        log_every=max(0, int(args.log_every)),
    )

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

    summary = {
        "config": config.to_dict(),
        "h_shape": [int(h.shape[0]), int(h.shape[1])],
        "h_nnz": int(h.nnz),
        "progress": {
            "episodes_completed": int(progress.episodes_completed),
            "total_updates": int(progress.total_updates),
            "mean_reward": float(progress.mean_reward()),
            "elapsed_sec": float(progress.elapsed_sec),
        },
        "artifacts": {
            "checkpoint_latest": str(latest_path),
            "checkpoint_final": str(final_path),
            "q_table_npy": str(q_table_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] training complete")
    print(f"[done] episodes_completed={progress.episodes_completed}")
    print(f"[done] total_updates={progress.total_updates}")
    print(f"[done] mean_reward={progress.mean_reward():.6f}")
    print(f"[done] artifacts: {checkpoint_dir}")


if __name__ == "__main__":
    _main()
