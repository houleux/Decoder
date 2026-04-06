from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_latest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def _collect(run_root: Path, kind: str) -> list[dict]:
    rows: list[dict] = []
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    for code in manifest.get("codes", []):
        file_name = "latest_berfer.json" if kind == "berfer" else "latest_messages.json"
        for row in _load_latest(run_root / code / "results" / file_name):
            row = dict(row)
            row["code"] = code
            rows.append(row)
    return rows


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


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate plots from latest stored eval tables")
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = run_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    berfer_rows = _collect(run_root, kind="berfer")
    msg_rows = _collect(run_root, kind="messages")

    saved = []
    saved.extend(_plot_berfer(berfer_rows, out_dir))
    saved.extend(_plot_messages(msg_rows, out_dir))

    if not saved:
        print("No latest eval tables found. Run eval jobs first.")
        return

    print("Saved plots:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    _main()
