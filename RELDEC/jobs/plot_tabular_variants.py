from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RELDEC.data_catalog import DataCatalog, ResultQuery

def _plot_berfer(rows: list[dict], out_dir: Path, code: str) -> list[Path]:
    saved: list[Path] = []
    if not rows:
        print(f"No rows found for {code}!")
        return saved

    # Only plot these specific methods in a consistent order
    target_methods = ["flooding", "reldec", "reldec_misq_local", "reldec_misq_global", "rel_delta"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("ber", "BER", axes[0]), ("fer", "FER", axes[1]), ("avg_messages", "Avg Messages", axes[2])]
    
    for metric_key, metric_title, ax in metrics:
        for method in target_methods:
            mr = [r for r in rows if str(r["method"]) == method]
            if not mr:
                continue
            
            # Group by SNR and take the latest run if duplicates exist
            snr_to_val = {}
            for r in mr:
                snr = float(r["snr_db"])
                val = float(r[metric_key])
                if metric_key in ("ber", "fer"):
                    val = max(val, 1e-12)
                # Assuming chronological order, last one wins, or we can just sort
                snr_to_val[snr] = val
                
            pts = sorted(snr_to_val.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=method)
            
        ax.set_title(f"{code.upper()} | {metric_title}")
        ax.set_xlabel("Eb/N0 (dB)")
        ax.set_ylabel(metric_title)
        if metric_key in ("ber", "fer"):
            ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(loc="best")
        
    fig.tight_layout()
    out = out_dir / f"{code}_tabular_variants.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    saved.append(out)
    return saved

def _main() -> None:
    catalog = DataCatalog()
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for code in ["mackay", "wran"]:
        print(f"Querying results for {code}...")
        query = ResultQuery(code=code)
        rows = catalog.query_evaluation_rows(query)
        saved = _plot_berfer(rows, out_dir, code)
        for path in saved:
            print(f"Saved plot: {path}")

if __name__ == "__main__":
    _main()
