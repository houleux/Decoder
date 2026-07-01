import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

def load_matrix(csv_path: str) -> tuple[sp.csr_matrix, float]:
    df = pd.read_csv(csv_path)
    m = int(df["row"].max() + 1)
    n = int(df["col"].max() + 1)
    h_csr = sp.csr_matrix(
        (np.ones(len(df), dtype=np.uint8), (df["row"], df["col"])),
        shape=(m, n),
        dtype=np.uint8
    )
    code_rate = 1.0 - (m / n)
    return h_csr, code_rate
from rl.decoder.engine import evaluate_snr_point, write_csv


def main():
    parser = argparse.ArgumentParser(description="Evaluate an LDPC decoder over an SNR sweep.")
    parser.add_argument("--matrix-csv",         required=True, help="Path to parity check matrix CSV")
    parser.add_argument("--method",             required=True, choices=["flooding", "round_robin", "random", "reldec", "dyna_reldec", "ave_res_q", "max_res_q", "factored_dqn", "rbl", "ave_rbl", "max_rbl", "llr_vec_ave_res", "llr_vec_ave_mi", "ave_llr_ave_res", "ave_llr_ave_mi", "tanh_vec_ave_res", "tanh_vec_ave_mi", "ave_tanh_ave_res", "ave_tanh_ave_mi"])
    parser.add_argument("--z",                  type=int, default=1, help="Cluster size (required for reldec)")
    parser.add_argument("--checkpoint",         default=None, help="Path to agent checkpoint JSON (required for reldec)")
    parser.add_argument("--snr-db",             required=True, nargs="+", type=float, help="Eb/N0 values in dB to evaluate")
    parser.add_argument("--i-max",              required=True, type=int, help="Max decoder iterations per frame")
    parser.add_argument("--target-frame-errors", type=int, default=100, help="Stop SNR point after this many frame errors")
    parser.add_argument("--max-frames",         type=int, default=100000, help="Maximum frames per SNR point")
    parser.add_argument("--workers",            type=int, default=1)
    parser.add_argument("--seed",               required=True, type=int)
    parser.add_argument("--output-csv",         required=True, help="Path to write results CSV")
    args = parser.parse_args()

    if args.method == "reldec" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --method reldec")

    h_csr, code_rate = load_matrix(args.matrix_csv)
    rng = np.random.default_rng(args.seed)

    results = []
    for snr_db in args.snr_db:
        print(f"Evaluating {args.method} at {snr_db:.2f} dB...")
        stats = evaluate_snr_point(
            h_csr=h_csr,
            method=args.method,
            z=args.z,
            checkpoint_path=args.checkpoint,
            ebn0_db=snr_db,
            code_rate=code_rate,
            i_max=args.i_max,
            target_frame_errors=args.target_frame_errors,
            max_frames=args.max_frames,
            rng=rng,
            n_workers=args.workers,
        )
        results.append((snr_db, stats))
        print(f"  frames={stats.frames}  BER={stats.ber:.5f}  FER={stats.fer:.5f}")

    write_csv(results, args.output_csv)
    print(f"Results written to {args.output_csv}")

if __name__ == "__main__":
    main()
