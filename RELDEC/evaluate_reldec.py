from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from RELDEC.experiments import EvaluationManifest, EvaluationSpec, ConfigLoader
from RELDEC.storage import RunStore, compute_config_hash
from RELDEC.registry import (
    supported_method_names,
    methods_requiring_q_table,
    methods_requiring_mi_tabular_q_table,
    methods_requiring_deep_checkpoint,
)

import numpy as np

from RELDEC.algorithms.reldec_deep import evaluate_deep_method, load_deep_decoder_from_checkpoint
from RELDEC.algorithms.reldec_deep import MiReldecBaselineDecoder
from RELDEC.algorithms.reldec_deep import MiTabularQDecoder, evaluate_mi_tabular_method
from RELDEC.algorithms.reldec_augmented import load_augmented_deep_decoder_from_checkpoint
from RELDEC.method_dispatcher import MethodDispatcher
from RELDEC.evaluation_router import evaluate_method_with_dispatcher
from RELDEC.algorithms.reldec_core import (
    THIS_DIR,
    ReldecDecoderSuite,
    evaluate_single_method,
    get_code_preset,
    load_parity_check_from_sparse_csv,
    load_q_table,
    nominal_code_rate,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RELDEC and baseline LDPC schedulers over SNR sweeps."
    )
    # Config file support
    parser.add_argument("--config", type=str, default=None, help="Load defaults from YAML/JSON config file")
    
    parser.add_argument("--code", choices=["ab", "wran", "mackay"], default="ab")
    parser.add_argument("--matrix-csv", type=str, default=None)
    parser.add_argument("--q-table", type=str, default=None)
    parser.add_argument("--mi-tabular-q-table", type=str, default=None)
    parser.add_argument("--deep-checkpoint", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Subset of: flooding random round_robin reldec deep_reldec_z1 "
            "mi_tabular_zx deep_reldec_zx mi_naive_zx mi_dqn_zx mi_tabular_zx "
            "augmented_max_avg_zx augmented_max_zx augmented_average_zx "
            "tabular_augmented_max_avg_zx tabular_augmented_max_zx tabular_augmented_average_zx"
        ),
    )
    parser.add_argument("--z", type=int, default=None, help="Cluster size for _zx methods")
    parser.add_argument("--mi-bins", type=int, default=21, help="Quantization level for MI state bins")
    parser.add_argument("--snr-db", nargs="+", type=float, default=None)
    parser.add_argument("--i-max", type=int, default=None)
    parser.add_argument("--code-rate", type=float, default=None)
    parser.add_argument("--target-frame-errors", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-codewords", action="store_true")
    parser.add_argument("--tabular-augmented-q-table", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes for frame evaluation (default: 1 = sequential)."
    )
    
    args = parser.parse_args()
    
    # Load config file if provided (CLI args override config file)
    if args.config:
        try:
            config_dict = ConfigLoader.load(args.config)
            config_args = ConfigLoader.evaluation_config_to_args(config_dict)
            
            # Apply config defaults only if CLI arg wasn't explicitly set
            for key, value in config_args.items():
                if hasattr(args, key):
                    current_val = getattr(args, key)
                    # If current value is the parser default and config has a value, use config
                    if current_val is None or (key not in {"code"} and current_val == parser.get_default(key)):
                        setattr(args, key, value)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load config file: {e}")
    
    return args


def _normalize_methods(args: argparse.Namespace) -> list[str]:
    if args.methods is None:
        methods = ["flooding", "random", "round_robin"]
        if args.q_table:
            methods.append("reldec")
        if args.mi_tabular_q_table:
            methods.append("mi_tabular_zx")
        if args.tabular_augmented_q_table:
            methods.append("tabular_augmented_max_avg_zx")
        return methods

    methods = [m.lower() for m in args.methods]
    valid = set(supported_method_names())
    for method in methods:
        if method not in valid:
            supported = ", ".join(sorted(valid))
            raise ValueError(f"Unknown method '{method}'. Supported methods: {supported}")
    return methods


def _write_csv(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _main() -> None:
    args = _parse_args()

    preset = get_code_preset(args.code)
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else preset.matrix_csv
    snr_db_values = list(args.snr_db) if args.snr_db else list(preset.eval_snr_db)
    i_max = int(args.i_max) if args.i_max is not None else int(preset.inference_i_max)

    methods = _normalize_methods(args)
    
    # Validate required checkpoints
    if methods_requiring_q_table(methods) and not args.q_table:
        raise ValueError("--q-table is required when evaluating RELDEC tabular method")
    if methods_requiring_mi_tabular_q_table(methods) and not args.mi_tabular_q_table:
        raise ValueError("--mi-tabular-q-table is required when evaluating MI tabular methods")
    from RELDEC.registry import methods_requiring_tabular_augmented_q_table
    if methods_requiring_tabular_augmented_q_table(methods) and not args.tabular_augmented_q_table:
        raise ValueError("--tabular-augmented-q-table is required when evaluating tabular augmented methods")
    if methods_requiring_deep_checkpoint(methods) and not args.deep_checkpoint:
        raise ValueError("--deep-checkpoint is required when evaluating deep learning methods")
    
    # Validate z parameter for dynamic-z methods
    dynamic_z_methods = [m for m in methods if m.endswith("_zx")]
    if dynamic_z_methods and args.z is None:
        raise ValueError(f"--z is required when evaluating _zx methods: {dynamic_z_methods}")

    h = load_parity_check_from_sparse_csv(matrix_csv)
    code_rate = args.code_rate if args.code_rate is not None else nominal_code_rate(h)

    # Initialize method dispatcher for all requested methods
    dispatcher = MethodDispatcher(
        matrix_csv=matrix_csv,
        h_csr=h,
        q_table_path=args.q_table,
        mi_tabular_q_table_path=args.mi_tabular_q_table,
        tabular_augmented_q_table_path=args.tabular_augmented_q_table,
        deep_checkpoint_path=args.deep_checkpoint,
        mi_bins=int(args.mi_bins),
        args_z=args.z,
    )

    rng = np.random.default_rng(args.seed)
    all_zero_only = not args.random_codewords

    run_identity = {
        "kind": "evaluation",
        "code": str(args.code),
        "matrix_csv": str(matrix_csv),
        "methods": sorted(methods),
        "parameters": {
            "z": args.z,
            "mi_bins": int(args.mi_bins),
            "i_max": int(i_max),
            "target_frame_errors": int(args.target_frame_errors),
            "max_frames": int(args.max_frames),
            "random_codewords": bool(args.random_codewords),
        },
    }
    run_hash = compute_config_hash(run_identity)
    run_id = f"eval_{run_hash[:12]}"

    output_dir = THIS_DIR / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "results.csv"
    output_json = Path(args.output_json) if args.output_json else output_dir / "results.json"
    manifest_path = output_dir / "evaluation_manifest.json"

    if output_json.exists() and output_csv.exists() and manifest_path.exists():
        print(f"[eval] run_id={run_id} already exists; reusing stored results")
        print(f"[eval] CSV: {output_csv}")
        print(f"[eval] JSON: {output_json}")
        return

    print(f"[eval] run_id={run_id}")
    print(f"[eval] code={args.code} matrix={matrix_csv}")
    print(f"[eval] H shape={h.shape} nnz={h.nnz} rate={code_rate:.6f} i_max={i_max}")
    print(f"[eval] methods={methods}")
    print(f"[eval] SNR points={snr_db_values}")
    print(
        "[eval] stop criteria: "
        f"target_frame_errors={args.target_frame_errors} max_frames={args.max_frames}"
    )

    rows: list[dict] = []
    suite = ReldecDecoderSuite(h)
    q_table_methods = {"reldec", "reldec_misq_global", "reldec_misq_local", "rel_delta",
                       "dyna_reldec", "dyna_reldelta", "dyna_mi", "dyna_midelta"}
    if q_table_methods & set(methods) and dispatcher.q_table is not None:
        suite.set_q_table(dispatcher.q_table)
    
    for snr_db in snr_db_values:
        print(f"[eval] snr={snr_db:.2f} dB")
        for method in methods:
            import time
            start_time = time.time()
            
            # Use evaluation router to dispatch to the right evaluation function
            stats = evaluate_method_with_dispatcher(
                dispatcher=dispatcher,
                method=method,
                snr_db=float(snr_db),
                code_rate=float(code_rate),
                i_max=i_max,
                target_frame_errors=int(args.target_frame_errors),
                max_frames=int(args.max_frames),
                rng=rng,
                all_zero_only=all_zero_only,
                suite=suite,
                n_workers=int(args.workers),
            )
            
            total_time = time.time() - start_time
            time_per_frame = total_time / stats.frames if stats.frames > 0 else 0

            row = stats.summary(snr_db=float(snr_db))
            row["code"] = args.code
            row["matrix_csv"] = str(matrix_csv)
            row["code_rate"] = float(code_rate)
            row["i_max"] = int(i_max)
            row["target_frame_errors"] = int(args.target_frame_errors)
            row["max_frames"] = int(args.max_frames)
            row["all_zero_only"] = bool(all_zero_only)
            row["total_time_sec"] = total_time
            row["time_per_frame_ms"] = time_per_frame * 1000.0
            rows.append(row)

            print(
                f"  - {method:11s} frames={row['frames']:7d} "
                f"FER={row['fer']:.6e} BER={row['ber']:.6e} "
                f"avg_msgs={row['avg_messages']:.2f} "
                f"time={total_time:.2f}s "
                f"({time_per_frame*1000.0:.2f}ms/frame)"
            )

    _write_csv(rows, output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "code": args.code,
            "matrix_csv": str(matrix_csv),
            "code_rate": float(code_rate),
            "methods": methods,
            "snr_db": snr_db_values,
            "i_max": i_max,
            "target_frame_errors": int(args.target_frame_errors),
            "max_frames": int(args.max_frames),
            "all_zero_only": bool(all_zero_only),
            "seed": int(args.seed),
        },
        "results": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = EvaluationManifest.create(
        run_id=run_id,
        experiment=EvaluationSpec(
            code=args.code,
            matrix_csv=str(matrix_csv),
            methods=methods,
            parameters={
                "z": args.z,
                "mi_bins": int(args.mi_bins),
                "i_max": int(i_max),
                "target_frame_errors": int(args.target_frame_errors),
                "max_frames": int(args.max_frames),
                "random_codewords": bool(args.random_codewords),
            },
        ),
        evaluation_config=payload["config"],
        artifacts={
            "results_csv": str(output_csv),
            "results_json": str(output_json),
            "config_hash": run_hash,
        },
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    RunStore(THIS_DIR / "runs").save_evaluation_run(manifest, output_dir)

    print(f"[done] wrote CSV: {output_csv}")
    print(f"[done] wrote JSON: {output_json}")
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    _main()
