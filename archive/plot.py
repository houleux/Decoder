import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot BER and FER from evaluation CSV files.")
    parser.add_argument("--csvs",   required=True, nargs="+", help="CSV files to plot")
    parser.add_argument("--labels", required=True, nargs="+", help="Legend labels (one per CSV)")
    parser.add_argument("--output", required=True, help="Path to save output plot (e.g. plot.png)")
    args = parser.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError("Number of --csvs must match number of --labels")

    fig, (ax_ber, ax_fer) = plt.subplots(1, 2, figsize=(12, 5))

    for csv_path, label in zip(args.csvs, args.labels):
        df = pd.read_csv(csv_path)
        snr_col = "ebn0_db" if "ebn0_db" in df.columns else "snr_db"

        df_ber = df[df["ber"] > 0]
        df_fer = df[df["fer"] > 0]

        ax_ber.semilogy(df_ber[snr_col], df_ber["ber"], marker="o", label=label)
        ax_fer.semilogy(df_fer[snr_col], df_fer["fer"], marker="o", label=label)

    for ax, ylabel, title in [
        (ax_ber, "Bit Error Rate (BER)", "BER vs Eb/N0"),
        (ax_fer, "Frame Error Rate (FER)", "FER vs Eb/N0"),
    ]:
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
