#!/usr/bin/env python3

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RELDEC_DIR = SCRIPT_DIR.parent
if str(RELDEC_DIR) not in sys.path:
    sys.path.insert(0, str(RELDEC_DIR))

from reldec_core import build_training_snr_schedule
from reldec_deep import (
    DeepTrainingCheckpoint,
    load_deep_training_checkpoint,
    save_deep_training_checkpoint,
)


def _extend_checkpoint_schedule(
    checkpoint_path: Path,
    target_total_episodes: int,
    backup_suffix: str,
) -> None:
    ck = load_deep_training_checkpoint(checkpoint_path)
    done = int(ck.progress.episodes_completed)
    old_sched = np.asarray(ck.snr_schedule_db, dtype=np.float64)

    if done > target_total_episodes:
        raise ValueError(
            f"episodes_completed={done} already exceeds target_total_episodes={target_total_episodes} "
            f"for {checkpoint_path}"
        )

    backup_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_before_extend_{backup_suffix}{checkpoint_path.suffix}"
    )
    shutil.copy2(checkpoint_path, backup_path)

    if old_sched.size >= target_total_episodes:
        new_sched = old_sched[:target_total_episodes]
        action = "trim"
    else:
        needed = target_total_episodes - old_sched.size
        snrs = tuple(float(x) for x in ck.config.train_snr_db)
        if not snrs:
            raise ValueError(f"Empty train_snr_db in checkpoint config for {checkpoint_path}")

        episodes_per_snr = (needed + len(snrs) - 1) // len(snrs)
        rng = np.random.default_rng(int(ck.config.seed) + 12345)
        extra = build_training_snr_schedule(snrs, episodes_per_snr, rng)[:needed]
        new_sched = np.concatenate([old_sched, extra])
        action = "extend"

    payload = DeepTrainingCheckpoint(
        config=ck.config,
        dqn_config=ck.dqn_config,
        progress=ck.progress,
        rng_state=ck.rng_state,
        snr_schedule_db=new_sched,
        global_step=ck.global_step,
        q_online_bytes=ck.q_online_bytes,
        q_target_bytes=ck.q_target_bytes,
        optimizer_bytes=ck.optimizer_bytes,
    )
    save_deep_training_checkpoint(checkpoint_path, payload)

    print(
        f"[{action}] {checkpoint_path} | episodes_completed={done} | "
        f"schedule: {old_sched.size} -> {new_sched.size} | backup={backup_path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend deep RELDEC checkpoint schedules to a target total episode count. "
            "Targets only non-constant epsilon folders by default."
        )
    )
    parser.add_argument(
        "--target-total-episodes",
        type=int,
        default=10000,
        help="Target total episodes after extension (default: 10000)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to Decoder repo root (default: inferred from script location)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    wran_root = (
        repo_root
        / "RELDEC"
        / "notebook_runs"
        / "continuous_reldec"
        / "active_run"
        / "wran"
    )

    targets = [
        wran_root / "checkpoints_deep_z1_1k" / "checkpoint_latest.npz",
        wran_root / "checkpoints_deep_z2_1k" / "checkpoint_latest.npz",
    ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    for ckpt in targets:
        if "eps06const" in str(ckpt):
            raise ValueError(f"Refusing constant-epsilon target: {ckpt}")
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    print(f"Repo root: {repo_root}")
    print(f"Target total episodes: {args.target_total_episodes}")
    print("Extending non-constant deep checkpoints only:")
    for ckpt in targets:
        print(f"  - {ckpt}")

    for ckpt in targets:
        _extend_checkpoint_schedule(
            checkpoint_path=ckpt,
            target_total_episodes=int(args.target_total_episodes),
            backup_suffix=stamp,
        )

    print("Done.")


if __name__ == "__main__":
    main()
