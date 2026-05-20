#!/usr/bin/env python3
"""Generate current-context documentation from RELDEC experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from context_sync import ContextSyncGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate context documentation from RELDEC experiments"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing experiment runs (default: runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Directory to write markdown files (default: docs)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate single full-context markdown file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for full context (if --full)",
    )
    
    args = parser.parse_args()
    
    generator = ContextSyncGenerator(runs_dir=args.runs_dir)
    
    if args.full:
        output_file = args.output or "CONTEXT.md"
        full_context = generator.generate_full_context()
        Path(output_file).write_text(full_context, encoding="utf-8")
        print(f"[context-sync] wrote full context: {output_file}")
    else:
        generator.write_context_bundle(output_dir=args.output_dir)
        print(f"[context-sync] wrote context bundle to {args.output_dir}/")
        print("[context-sync] generated files:")
        print("  - METHODS.md (method catalog)")
        print("  - POLICIES.md (training policies)")
        print("  - RUNS.md (experiment history)")
        print("  - STATUS.md (system status)")


if __name__ == "__main__":
    main()
