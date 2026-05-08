#!/usr/bin/env python3
"""Smoke test training for global-state RL methods on WRAN and Mackay codes."""

import numpy as np
import scipy.sparse as sp
from pathlib import Path

from reldec_core import load_parity_check_from_sparse_csv, bpsk_awgn_llr, nominal_code_rate, MethodStats
from reldec_global_mdp import (
    FullStateBinaryTabularTrainer,
    FullStateBinaryDeepTrainer,
    FullStateLLRDeepTrainer,
)
import csv
import json

THIS_DIR = Path(__file__).parent
RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def evaluate_and_save(trainer, code_name: str, method_name: str, config: dict):
    print(f"\nEvaluating {method_name} on {code_name}...")
    stats_list = []
    
    rng = np.random.default_rng(42)
    for snr_db in config["snr_db"]:
        stats = MethodStats(method=method_name, n=trainer.n)
        target_fe = 60
        max_frames = 3000
        
        while stats.frame_errors < target_fe and stats.frames < max_frames:
            tx_bits = np.zeros(trainer.n, dtype=np.uint8)
            llr = bpsk_awgn_llr(tx_bits, snr_db, config["code_rate"], rng)
            decoded = trainer.decode(llr, i_max=config["l_max"], rng=rng)
            stats.update(tx_bits, decoded)
            
        row = stats.summary(snr_db=snr_db)
        row["code"] = code_name
        row["matrix_csv"] = str(config["matrix"])
        row["code_rate"] = config["code_rate"]
        row["i_max"] = config["l_max"]
        row["target_frame_errors"] = target_fe
        row["max_frames"] = max_frames
        row["all_zero_only"] = True
        stats_list.append(row)
        print(f"  SNR {snr_db}dB: BER={row['ber']:.4e} FER={row['fer']:.4e} Frames={row['frames']}")
        
    csv_path = RESULTS_DIR / f"eval_global_mdp_{method_name}_{code_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
        writer.writeheader()
        writer.writerows(stats_list)
    print(f"Saved evaluation to {csv_path}")


# ============================================================================
# CONFIG
# ============================================================================

SMOKE_CONFIG = {
    "wran": {
        "matrix": THIS_DIR / "matrices" / "WRAN_irreg_384_256.csv",
        "snr_db": (1.0, 2.0, 3.0, 4.0, 5.0, 5.5),
        "episodes_per_snr": 50,  # Full smoke run: 50 episodes/SNR
        "l_max": 10,  # More realistic than 5
        "code_rate": 2/3,  # 256/384
    },
    "mackay": {
        "matrix": THIS_DIR / "matrices" / "H_Mackay_96_48.csv",
        "snr_db": (0.5, 1.0, 1.5, 2.0, 2.5),
        "episodes_per_snr": 50,  # Full smoke run: 50 episodes/SNR
        "l_max": 10,  # More realistic than 5
        "code_rate": 0.5,  # 48/96
    },
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_tabular_method(code_name: str, config: dict):
    """Train FullStateBinaryTabularTrainer."""
    print(f"\n{'='*70}")
    print(f"Training FullStateBinaryTabular on {code_name.upper()}")
    print(f"{'='*70}")
    
    h = load_parity_check_from_sparse_csv(config["matrix"])
    snr_list = config["snr_db"]
    episodes_per_snr = config["episodes_per_snr"]
    l_max = config["l_max"]
    code_rate = config["code_rate"]
    
    print(f"Code: {config['matrix'].name}")
    print(f"  n={h.shape[1]}, m={h.shape[0]}, rate={code_rate:.3f}")
    print(f"SNRs: {snr_list}")
    print(f"Episodes/SNR: {episodes_per_snr}, l_max: {l_max}")
    
    trainer = FullStateBinaryTabularTrainer(
        h_csr=h,
        alpha=0.1,
        beta=0.9,
        epsilon=0.6,
        l_max=l_max,
        cluster_size=2,
    )
    
    rng = np.random.default_rng(42)
    total_episodes = len(snr_list) * episodes_per_snr
    
    for snr_idx, snr_db in enumerate(snr_list):
        print(f"\nSNR {snr_idx+1}/{len(snr_list)}: {snr_db} dB")
        
        for ep in range(episodes_per_snr):
            # Generate random LLR channel
            n = h.shape[1]
            tx_bits = rng.integers(0, 2, size=n)
            awgn_llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
            
            # Train episode
            episode_reward = trainer.train_episode(awgn_llr, rng)
            
            if (ep + 1) % max(1, episodes_per_snr // 5) == 0 or ep == episodes_per_snr - 1:
                print(f"  Episode {ep+1:3d}/{episodes_per_snr}: reward={episode_reward:.3f}")
    
    print(f"\n✓ Tabular training complete. Q-table size: {len(trainer.q_table)}")
    return trainer


def train_binary_deep_method(code_name: str, config: dict):
    """Train FullStateBinaryDeepTrainer."""
    print(f"\n{'='*70}")
    print(f"Training FullStateBinaryDeep on {code_name.upper()}")
    print(f"{'='*70}")
    
    h = load_parity_check_from_sparse_csv(config["matrix"])
    snr_list = config["snr_db"]
    episodes_per_snr = config["episodes_per_snr"]
    l_max = config["l_max"]
    code_rate = config["code_rate"]
    
    print(f"Code: {config['matrix'].name}")
    print(f"  n={h.shape[1]}, m={h.shape[0]}, rate={code_rate:.3f}")
    print(f"SNRs: {snr_list}")
    print(f"Episodes/SNR: {episodes_per_snr}, l_max: {l_max}")
    
    trainer = FullStateBinaryDeepTrainer(
        h_csr=h,
        alpha=1e-3,  # DQN learning rate
        beta=0.9,
        epsilon=0.6,
        l_max=l_max,
        cluster_size=2,
        hidden_dim=128,
        batch_size=32,
        device="cpu",
    )
    
    rng = np.random.default_rng(42)
    total_episodes = len(snr_list) * episodes_per_snr
    
    for snr_idx, snr_db in enumerate(snr_list):
        print(f"\nSNR {snr_idx+1}/{len(snr_list)}: {snr_db} dB")
        
        total_loss = 0.0
        for ep in range(episodes_per_snr):
            # Generate random LLR channel
            n = h.shape[1]
            tx_bits = rng.integers(0, 2, size=n)
            awgn_llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
            
            # Train episode
            episode_reward, episode_loss = trainer.train_episode(awgn_llr, rng)
            total_loss += episode_loss
            
            if (ep + 1) % max(1, episodes_per_snr // 5) == 0 or ep == episodes_per_snr - 1:
                avg_loss = total_loss / (ep + 1) if ep > 0 else 0.0
                print(f"  Episode {ep+1:3d}/{episodes_per_snr}: reward={episode_reward:.3f}, avg_loss={avg_loss:.4f}")
    
    print(f"\n✓ Binary Deep training complete. Global step: {trainer.global_step}")
    return trainer


def train_llr_deep_method(code_name: str, config: dict):
    """Train FullStateLLRDeepTrainer."""
    print(f"\n{'='*70}")
    print(f"Training FullStateLLRDeep on {code_name.upper()}")
    print(f"{'='*70}")
    
    h = load_parity_check_from_sparse_csv(config["matrix"])
    snr_list = config["snr_db"]
    episodes_per_snr = config["episodes_per_snr"]
    l_max = config["l_max"]
    code_rate = config["code_rate"]
    
    print(f"Code: {config['matrix'].name}")
    print(f"  n={h.shape[1]}, m={h.shape[0]}, rate={code_rate:.3f}")
    print(f"SNRs: {snr_list}")
    print(f"Episodes/SNR: {episodes_per_snr}, l_max: {l_max}")
    
    trainer = FullStateLLRDeepTrainer(
        h_csr=h,
        alpha=1e-3,
        beta=0.9,
        epsilon=0.6,
        l_max=l_max,
        cluster_size=2,
        hidden_dim=128,
        batch_size=32,
        device="cpu",
    )
    
    rng = np.random.default_rng(42)
    
    for snr_idx, snr_db in enumerate(snr_list):
        print(f"\nSNR {snr_idx+1}/{len(snr_list)}: {snr_db} dB")
        
        total_loss = 0.0
        for ep in range(episodes_per_snr):
            # Generate random LLR channel
            n = h.shape[1]
            tx_bits = rng.integers(0, 2, size=n)
            awgn_llr = bpsk_awgn_llr(tx_bits, snr_db, code_rate, rng)
            
            # Train episode
            episode_reward, episode_loss = trainer.train_episode(awgn_llr, rng)
            total_loss += episode_loss
            
            if (ep + 1) % 10 == 0:
                avg_loss = total_loss / (ep + 1) if ep > 0 else 0.0
                print(f"  Episode {ep+1:3d}/{episodes_per_snr}: reward={episode_reward:.3f}, avg_loss={avg_loss:.4f}")
    
    print(f"\n✓ LLR Deep training complete. Global step: {trainer.global_step}")
    return trainer


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SMOKE TEST: Global-State RL Methods")
    print("="*70)
    
    # ========== WRAN ==========
    print("\n" + "▶ "*35)
    print("WRAN Code Training")
    wran_cfg = SMOKE_CONFIG["wran"]
    wran_tabular = train_tabular_method("wran", wran_cfg)
    evaluate_and_save(wran_tabular, "wran", "full_state_tabular_z", wran_cfg)
    
    wran_binary_deep = train_binary_deep_method("wran", wran_cfg)
    evaluate_and_save(wran_binary_deep, "wran", "full_binary_state_deep_z", wran_cfg)
    
    wran_llr_deep = train_llr_deep_method("wran", wran_cfg)
    evaluate_and_save(wran_llr_deep, "wran", "full_llr_state_deep_z", wran_cfg)
    
    # ========== MACKAY ==========
    print("\n" + "▶ "*35)
    print("Mackay Code Training")
    mackay_cfg = SMOKE_CONFIG["mackay"]
    mackay_tabular = train_tabular_method("mackay", mackay_cfg)
    evaluate_and_save(mackay_tabular, "mackay", "full_state_tabular_z", mackay_cfg)
    
    mackay_binary_deep = train_binary_deep_method("mackay", mackay_cfg)
    evaluate_and_save(mackay_binary_deep, "mackay", "full_binary_state_deep_z", mackay_cfg)
    
    mackay_llr_deep = train_llr_deep_method("mackay", mackay_cfg)
    evaluate_and_save(mackay_llr_deep, "mackay", "full_llr_state_deep_z", mackay_cfg)
    
    print("\n" + "="*70)
    print("✓ All smoke test training runs complete!")
    print("="*70)