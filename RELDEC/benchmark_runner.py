#!/usr/bin/env python3
"""
Phase 6: Benchmark Runner - Execute normalized benchmark suite

This script runs the complete benchmark suite across all 16 methods and 5 matrices.
It manages training and evaluation according to normalized budgets.

Status: READY (No runs executed yet)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMatrix:
    """Matrix configuration for benchmark"""
    code: str
    name: str
    description: str
    file: Optional[str] = None
    

@dataclass
class BenchmarkMethod:
    """Method configuration for benchmark"""
    name: str
    family: str
    config_file: str
    requires_training: bool = True
    budget_episodes: int = 0
    device: str = "cpu"


@dataclass
class BenchmarkRun:
    """Single benchmark run specification"""
    run_id: str
    method: str
    matrix: str
    config_path: str
    budget_episodes: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class BenchmarkSuite:
    """Manage benchmark suite execution"""
    
    # Defined matrices for Phase 6
    MATRICES = {
        "ab": BenchmarkMatrix(
            code="ab",
            name="H_AB_3_7_196",
            description="Regular (3,7) LDPC - 196 variables, 98 checks",
        ),
        "ab500": BenchmarkMatrix(
            code="ab500",
            name="H_AB_LDPC_500",
            description="Regular (5,0) LDPC - 500 variables, 250 checks",
        ),
        "mackay": BenchmarkMatrix(
            code="mackay",
            name="H_Mackay_96_48",
            description="Classical irregular - 96 variables, 48 checks",
        ),
        "wran": BenchmarkMatrix(
            code="wran",
            name="WRAN_irreg_384_256",
            description="IEEE 802.16 WiMAX - 256 variables, ~128 checks",
        ),
        "nr520": BenchmarkMatrix(
            code="nr520",
            name="H_5GNR_520_100",
            description="5G NR moderate size - 520 variables, 100 checks",
        ),
    }
    
    # Defined methods for Phase 6
    METHODS = {
        # Baseline methods (no training)
        "flooding": BenchmarkMethod(
            name="flooding",
            family="baseline",
            config_file="baseline_methods.yaml",
            requires_training=False,
            device="cpu",
        ),
        "random": BenchmarkMethod(
            name="random",
            family="baseline",
            config_file="baseline_methods.yaml",
            requires_training=False,
            device="cpu",
        ),
        "round_robin": BenchmarkMethod(
            name="round_robin",
            family="baseline",
            config_file="baseline_methods.yaml",
            requires_training=False,
            device="cpu",
        ),
        # Tabular methods (15k budget = 2.5k per SNR x 6 SNR points)
        "reldec": BenchmarkMethod(
            name="reldec",
            family="tabular",
            config_file="tabular_reldec.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cpu",
        ),
        "mi_tabular_zx": BenchmarkMethod(
            name="mi_tabular_zx",
            family="mi_tabular",
            config_file="tabular_mi_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cpu",
        ),
        # Deep methods (15k budget = 2.5k per SNR x 6 SNR points)
        "deep_reldec_zx": BenchmarkMethod(
            name="deep_reldec_zx",
            family="deep",
            config_file="deep_reldec_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cuda",
        ),
        "mi_dqn_zx": BenchmarkMethod(
            name="mi_dqn_zx",
            family="mi_dqn",
            config_file="mi_dqn_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cuda",
        ),
        "augmented_max_avg_zx": BenchmarkMethod(
            name="augmented_max_avg_zx",
            family="augmented",
            config_file="augmented_max_avg_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cuda",
        ),
        "augmented_max_zx": BenchmarkMethod(
            name="augmented_max_zx",
            family="augmented",
            config_file="augmented_max_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cuda",
        ),
        "augmented_average_zx": BenchmarkMethod(
            name="augmented_average_zx",
            family="augmented",
            config_file="augmented_average_zx.yaml",
            requires_training=True,
            budget_episodes=15000,
            device="cuda",
        ),
    }
    
    def __init__(self, config_dir: Path, runs_dir: Path):
        self.config_dir = Path(config_dir)
        self.runs_dir = Path(runs_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plan_runs(self) -> list[BenchmarkRun]:
        """Generate all benchmark runs"""
        runs = []
        
        for matrix_key, matrix in self.MATRICES.items():
            for method_key, method in self.METHODS.items():
                run_id = f"benchmark_{method.family}_{method_key}_{matrix_key}_{self.timestamp}"
                config_path = self.config_dir / "benchmark" / method.config_file
                
                run = BenchmarkRun(
                    run_id=run_id,
                    method=method_key,
                    matrix=matrix_key,
                    config_path=str(config_path),
                    budget_episodes=method.budget_episodes,
                )
                runs.append(run)
        
        return runs
    
    def print_plan(self, runs: list[BenchmarkRun]) -> None:
        """Print benchmark plan"""
        print("\n" + "="*80)
        print("PHASE 6: BENCHMARK NORMALIZATION PLAN")
        print("="*80)
        print(f"\nMatrices: {len(self.MATRICES)}")
        print(f"Methods:  {len(self.METHODS)}")
        print(f"Total runs: {len(runs)}")
        
        print("\n" + "-"*80)
        print("MATRICES:")
        print("-"*80)
        for code, matrix in self.MATRICES.items():
            print(f"  {code:10s} - {matrix.name:30s} {matrix.description}")
        
        print("\n" + "-"*80)
        print("METHODS BY FAMILY:")
        print("-"*80)
        families = {}
        for method_key, method in self.METHODS.items():
            if method.family not in families:
                families[method.family] = []
            families[method.family].append((method_key, method))
        
        for family in sorted(families.keys()):
            methods = families[family]
            print(f"\n  {family.upper()}:")
            for method_key, method in sorted(methods):
                budget_str = f"{method.budget_episodes:,} eps" if method.budget_episodes else "no budget"
                device = method.device
                training = "training" if method.requires_training else "eval only"
                print(f"    {method_key:30s} [{device:5s}] {training:15s} {budget_str}")
        
        print("\n" + "-"*80)
        print("BUDGET SUMMARY:")
        print("-"*80)
        budgets = {}
        for method in self.METHODS.values():
            if method.family not in budgets:
                budgets[method.family] = 0
            budgets[method.family] += method.budget_episodes * len(self.MATRICES)
        
        total_episodes = 0
        for family in sorted(budgets.keys()):
            episodes = budgets[family]
            print(f"  {family:15s}: {episodes:>12,} total episodes")
            total_episodes += episodes
        
        print(f"  {'TOTAL':15s}: {total_episodes:>12,} total episodes")
        
        # Estimate time and space
        eps_per_sec = 1000  # rough estimate
        hours = total_episodes / (eps_per_sec * 3600)
        space_gb = total_episodes * 10 / 1e9  # ~10 bytes per episode
        
        print(f"\n  Estimated time (GPU): {hours:.1f} hours")
        print(f"  Estimated space: {space_gb:.1f} GB")
        
        print("\n" + "="*80)
        print("STATUS: Ready for execution (no runs performed yet)")
        print("="*80 + "\n")
    
    def save_plan_json(self, runs: list[BenchmarkRun], output_file: Path) -> None:
        """Save plan as JSON"""
        plan = {
            "timestamp": self.timestamp,
            "total_runs": len(runs),
            "matrices": {k: asdict(v) for k, v in self.MATRICES.items()},
            "methods": {k: asdict(v) for k, v in self.METHODS.items()},
            "runs": [asdict(r) for r in runs],
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(plan, f, indent=2)
        
        logger.info(f"Plan saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Benchmark Normalization Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print plan only
  python benchmark_runner.py --print-plan
  
  # Save plan as JSON
  python benchmark_runner.py --save-plan phase6_plan.json
  
  # Execute benchmark (requires GPU and ~1 week)
  python benchmark_runner.py --execute
        """,
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing benchmark configs",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent.parent / "runs",
        help="Directory to store benchmark runs",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print benchmark plan and exit",
    )
    parser.add_argument(
        "--save-plan",
        type=Path,
        help="Save plan as JSON to file",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute benchmark suite",
    )
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(args.config_dir, args.runs_dir)
    runs = suite.plan_runs()
    
    if args.print_plan or not (args.save_plan or args.execute):
        suite.print_plan(runs)
    
    if args.save_plan:
        suite.save_plan_json(runs, args.save_plan)
    
    if args.execute:
        logger.error("Benchmark execution not yet implemented")
        logger.error("To execute, call train_reldec.py and evaluate_reldec.py with benchmark configs")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
