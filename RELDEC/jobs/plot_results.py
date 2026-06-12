from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RELDEC.data_catalog import DataCatalog, ResultQuery


def _query_rows(
    *,
    code: str | None = None,
    method: str | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
    snr_min: float | None = None,
    snr_max: float | None = None,
) -> list[dict]:
    catalog = DataCatalog()
    query = ResultQuery(
        code=code,
        method=method,
        run_id=run_id,
        config_hash=config_hash,
        snr_min=snr_min,
        snr_max=snr_max,
    )
    return catalog.query_evaluation_rows(query)


def _plot_berfer(rows: list[dict], out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if not rows:
        return saved

    codes = sorted({str(r["code"]) for r in rows})
    methods = sorted({str(r["method"]) for r in rows})

    for code in codes:
        code_rows = [r for r in rows if str(r["code"]) == code]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for metric_key, metric_title, ax in [("ber", "BER", axes[0]), ("fer", "FER", axes[1])]:
            for method in methods:
                mr = [r for r in code_rows if str(r["method"]) == method]
                if not mr:
                    continue
                pts = sorted((float(r["snr_db"]), max(float(r[metric_key]), 1e-12)) for r in mr)
                ax.semilogy([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=method)
            ax.set_title(f"{code.upper()} | {metric_title}")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel(metric_title)
            ax.grid(True, which="both", linestyle="--", alpha=0.35)
        axes[1].legend(loc="best")
        fig.tight_layout()
        out = out_dir / f"{code}_ber_fer.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(out)

    return saved


def _plot_messages(rows: list[dict], out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    if not rows:
        return saved

    codes = sorted({str(r["code"]) for r in rows})
    methods = sorted({str(r["method"]) for r in rows})

    for code in codes:
        code_rows = [r for r in rows if str(r["code"]) == code]
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
        for method in methods:
            mr = [r for r in code_rows if str(r["method"]) == method]
            if not mr:
                continue
            pts = sorted((float(r["snr_db"]), float(r["avg_messages"])) for r in mr)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=method)
        ax.set_title(f"{code.upper()} | Avg CN->VN Messages")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel("Avg CN->VN Messages")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        out = out_dir / f"{code}_messages.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(out)

    return saved


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate plots from stored RELDEC evaluation results")
    ap.add_argument("--code", type=str, default=None)
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--config-hash", type=str, default=None)
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--snr-max", type=float, default=None)
    ap.add_argument("--kind", choices=["berfer", "messages", "both"], default="both")
    ap.add_argument("--output-dir", type=str, default=None)
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()

    rows = _query_rows(
        code=args.code,
        method=args.method,
        run_id=args.run_id,
        config_hash=args.config_hash,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if args.kind in {"berfer", "both"}:
        saved.extend(_plot_berfer(rows, out_dir))
    if args.kind in {"messages", "both"}:
        saved.extend(_plot_messages(rows, out_dir))

    if not saved:
        print("No matching evaluation rows found. Run eval jobs or relax the filters.")
        return

    print("Saved plots:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    _main()
