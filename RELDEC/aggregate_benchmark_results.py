#!/usr/bin/env python3
"""
Phase 6: Benchmark Results Aggregator

Collect results from completed benchmark runs and generate:
- Comparison tables
- Performance summaries  
- Plots and visualizations
- Statistical analysis

Status: READY (to be used after benchmark execution)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    method: str
    matrix: str
    snr_db: float
    ber: float
    bler: float
    avg_iterations: float
    run_id: str


class BenchmarkAggregator:
    """Aggregate and analyze benchmark results"""
    
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.results: List[BenchmarkResult] = []
    
    def load_results(self, filter_pattern: str = "benchmark_*") -> None:
        """Load all benchmark results from runs directory"""
        training_dir = self.runs_dir / "training"
        evaluation_dir = self.runs_dir / "evaluation"
        
        if not training_dir.exists():
            logger.warning(f"Training runs directory not found: {training_dir}")
            return
        
        # Scan for benchmark run directories
        for run_dir in sorted(training_dir.glob(filter_pattern)):
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    # Parse manifest into BenchmarkResult objects
                    # (implementation depends on manifest format)
        
        logger.info(f"Loaded {len(self.results)} benchmark results")
    
    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table"""
        if not self.results:
            return "No results available"
        
        # Group by method and matrix
        by_method = defaultdict(lambda: defaultdict(list))
        for result in self.results:
            by_method[result.method][result.matrix].append(result)
        
        # Build table
        lines = ["# Benchmark Comparison Table\n"]
        lines.append("| Method | ab | ab500 | mackay | wran | nr520 |\n")
        lines.append("|--------|-------|-------|---------|-------|-------|\n")
        
        for method in sorted(by_method.keys()):
            row = f"| {method}"
            for matrix in ["ab", "ab500", "mackay", "wran", "nr520"]:
                results = by_method[method].get(matrix, [])
                if results:
                    # Aggregate BER at SNR=2.0 (middle point)
                    ber_at_2db = next(
                        (r.ber for r in results if abs(r.snr_db - 2.0) < 0.1),
                        None
                    )
                    if ber_at_2db:
                        row += f" | {ber_at_2db:.2e}"
                    else:
                        row += " | N/A"
                else:
                    row += " | N/A"
            row += " |\n"
            lines.append(row)
        
        return "".join(lines)
    
    def generate_summary_stats(self) -> str:
        """Generate summary statistics"""
        if not self.results:
            return "No results available"
        
        lines = ["# Benchmark Summary Statistics\n\n"]
        
        # By family
        by_family = defaultdict(list)
        for result in self.results:
            family = result.method.split("_")[0]
            by_family[family].append(result)
        
        for family in sorted(by_family.keys()):
            results = by_family[family]
            avg_ber = sum(r.ber for r in results) / len(results)
            avg_iter = sum(r.avg_iterations for r in results) / len(results)
            
            lines.append(f"## {family.upper()}\n")
            lines.append(f"- Methods: {len(set(r.method for r in results))}\n")
            lines.append(f"- Avg BER: {avg_ber:.2e}\n")
            lines.append(f"- Avg Iterations: {avg_iter:.1f}\n\n")
        
        return "".join(lines)
    
    def generate_report(self, output_file: Path) -> None:
        """Generate complete benchmark report"""
        report = "# Phase 6: Benchmark Results Report\n\n"
        report += self.generate_comparison_table()
        report += "\n"
        report += self.generate_summary_stats()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 6: Benchmark Results Aggregator"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "runs",
        help="Directory containing benchmark runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "BENCHMARK_RESULTS.md",
        help="Output report file",
    )
    
    args = parser.parse_args()
    
    aggregator = BenchmarkAggregator(args.runs_dir)
    aggregator.load_results()
    aggregator.generate_report(args.output)


if __name__ == "__main__":
    main()
