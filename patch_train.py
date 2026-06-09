import re

with open("RELDEC/train_reldec.py", "r") as f:
    code = f.read()

# Add _is_tabular_policy helper
helper = """
def _is_tabular_policy(policy_type: str) -> bool:
    return policy_type in {"tabular", "mi_tabular_z2", "mi_tabular_zx", "reldec_misq_local", "reldec_misq_global", "rel_delta"} or policy_type.startswith("tabular_augmented_")
"""

code = code.replace("def _parse_args() -> argparse.Namespace:", helper + "\n\ndef _parse_args() -> argparse.Namespace:")

# Replace explicit checks with the helper
code = re.sub(r'args\.policy_type in \{"tabular", "mi_tabular_z2", "mi_tabular_zx"\} or args\.policy_type\.startswith\("tabular_augmented_"\)', '_is_tabular_policy(args.policy_type)', code)
code = re.sub(r'args\.policy_type in \{"tabular", "mi_tabular_z2", "mi_tabular_zx", "reldec_misq_local", "reldec_misq_global", "rel_delta"\} or args\.policy_type\.startswith\("tabular_augmented_"\)', '_is_tabular_policy(args.policy_type)', code)
code = re.sub(r'args\.policy_type in \{"tabular", "mi_tabular_z2", "mi_tabular_zx"\}', '_is_tabular_policy(args.policy_type)', code)

# Replace the auto_resume block
old_auto_resume = """        if _is_tabular_policy(args.policy_type):
            checkpoint = load_training_checkpoint(latest_path)
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
                    cluster_size=cluster_size,
                    mi_bins=int(args.mi_bins),
                    q_table=checkpoint.q_table,
                )
            rng = np.random.default_rng()"""

new_auto_resume = """        if _is_tabular_policy(args.policy_type):
            checkpoint = load_training_checkpoint(latest_path)
            config = checkpoint.config
            progress = checkpoint.progress
            snr_schedule_db = checkpoint.snr_schedule_db
            h = load_parity_check_from_sparse_csv(config.matrix_csv)
            trainer = TrainerFactory.create_tabular_trainer(
                h_csr=h,
                config=config,
                policy_type=args.policy_type,
                mi_bins=int(args.mi_bins),
            )
            trainer.q_table = checkpoint.q_table
            rng = np.random.default_rng()"""

code = code.replace(old_auto_resume, new_auto_resume)

with open("RELDEC/train_reldec.py", "w") as f:
    f.write(code)
