from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from reldec_deep import evaluate_deep_method, load_deep_decoder_from_checkpoint
from reldec_deep import MiReldecBaselineDecoder
from reldec_core import (
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
    parser.add_argument("--code", choices=["ab", "wran"], default="ab")
    parser.add_argument("--matrix-csv", type=str, default=None)
    parser.add_argument("--q-table", type=str, default=None)
    parser.add_argument("--deep-checkpoint", type=str, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of: flooding random round_robin reldec deep_reldec_z1 deep_reldec_z2 mi_naive_z2 mi_dqn_z2",
    )
    parser.add_argument("--snr-db", nargs="+", type=float, default=None)
    parser.add_argument("--i-max", type=int, default=None)
    parser.add_argument("--code-rate", type=float, default=None)
    parser.add_argument("--target-frame-errors", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-codewords", action="store_true")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def _normalize_methods(args: argparse.Namespace) -> list[str]:
    if args.methods is None:
        methods = ["flooding", "random", "round_robin"]
        if args.q_table:
            methods.append("reldec")
        if args.deep_checkpoint:
            methods.append("deep_reldec_z2")
        return methods

    methods = [m.lower() for m in args.methods]
    valid = {
        "flooding",
        "random",
        "round_robin",
        "reldec",
        "deep_reldec_z1",
        "deep_reldec_z2",
        "mi_naive_z2",
        "mi_dqn_z2",
    }
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
    if "reldec" in methods and not args.q_table:
        raise ValueError("--q-table is required when evaluating method 'reldec'")
    if any(m.startswith("deep_reldec_") for m in methods) and not args.deep_checkpoint:
        raise ValueError("--deep-checkpoint is required when evaluating deep RELDEC methods")

    h = load_parity_check_from_sparse_csv(matrix_csv)
    code_rate = args.code_rate if args.code_rate is not None else nominal_code_rate(h)

    suite = ReldecDecoderSuite(h)
    if "reldec" in methods:
        suite.set_q_table(load_q_table(args.q_table))

    deep_decoders = {}
    if "deep_reldec_z1" in methods:
        deep_decoders["deep_reldec_z1"] = load_deep_decoder_from_checkpoint(
            checkpoint_path=args.deep_checkpoint,
            matrix_csv=matrix_csv,
            expected_policy_label="deep_z1",
        )
    if "deep_reldec_z2" in methods:
        deep_decoders["deep_reldec_z2"] = load_deep_decoder_from_checkpoint(
            checkpoint_path=args.deep_checkpoint,
            matrix_csv=matrix_csv,
            expected_policy_label="deep_z2",
        )
    if "mi_dqn_z2" in methods:
        deep_decoders["mi_dqn_z2"] = load_deep_decoder_from_checkpoint(
            checkpoint_path=args.deep_checkpoint,
            matrix_csv=matrix_csv,
            expected_policy_label="mi_dqn_z2",
        )

    mi_naive_decoder = MiReldecBaselineDecoder(load_parity_check_from_sparse_csv(matrix_csv), cluster_size=2)

    rng = np.random.default_rng(args.seed)
    all_zero_only = not args.random_codewords

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else THIS_DIR / "results" / f"eval_{args.code}_{ts}.csv"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_csv.with_suffix(".json")
    )

    print(f"[eval] code={args.code} matrix={matrix_csv}")
    print(f"[eval] H shape={h.shape} nnz={h.nnz} rate={code_rate:.6f} i_max={i_max}")
    print(f"[eval] methods={methods}")
    print(f"[eval] SNR points={snr_db_values}")
    print(
        "[eval] stop criteria: "
        f"target_frame_errors={args.target_frame_errors} max_frames={args.max_frames}"
    )

    rows: list[dict] = []
    for snr_db in snr_db_values:
        print(f"[eval] snr={snr_db:.2f} dB")
        for method in methods:
            stats = evaluate_single_method(
                suite=suite,
                method=method,
                snr_db=float(snr_db),
                code_rate=float(code_rate),
                i_max=i_max,
                target_frame_errors=int(args.target_frame_errors),
                max_frames=int(args.max_frames),
                rng=rng,
                all_zero_only=all_zero_only,
            ) if method in {"flooding", "random", "round_robin", "reldec"} else evaluate_deep_method(
                decoder=deep_decoders[method],
                snr_db=float(snr_db),
                code_rate=float(code_rate),
                i_max=i_max,
                target_frame_errors=int(args.target_frame_errors),
                max_frames=int(args.max_frames),
                rng=rng,
                all_zero_only=all_zero_only,
                method_name=method,
            ) if method == "mi_dqn_z2" else mi_naive_decoder.evaluate(
                snr_db=float(snr_db),
                code_rate=float(code_rate),
                i_max=i_max,
                target_frame_errors=int(args.target_frame_errors),
                max_frames=int(args.max_frames),
                rng=rng,
                all_zero_only=all_zero_only,
                method_name=method,
            )

            row = stats.summary(snr_db=float(snr_db))
            row["code"] = args.code
            row["matrix_csv"] = str(matrix_csv)
            row["code_rate"] = float(code_rate)
            row["i_max"] = int(i_max)
            row["target_frame_errors"] = int(args.target_frame_errors)
            row["max_frames"] = int(args.max_frames)
            row["all_zero_only"] = bool(all_zero_only)
            rows.append(row)

            print(
                f"  - {method:11s} frames={row['frames']:7d} "
                f"FER={row['fer']:.6e} BER={row['ber']:.6e} "
                f"avg_msgs={row['avg_messages']:.2f}"
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

    print(f"[done] wrote CSV: {output_csv}")
    print(f"[done] wrote JSON: {output_json}")


if __name__ == "__main__":
    _main()
