"""Context synchronization - generate markdown documentation from experiments."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from storage import RunStore
from registry import METHOD_CATALOG, TRAINING_POLICY_CATALOG


class ContextSyncGenerator:
    """Generate current-context markdown files for assistant-facing documentation."""

    def __init__(self, runs_dir: str | Path = "runs"):
        """Initialize the context sync generator.
        
        Args:
            runs_dir: Directory where runs are stored
        """
        self.store = RunStore(runs_dir)
        self.timestamp = datetime.now().isoformat()

    def generate_method_catalog_md(self) -> str:
        """Generate markdown documentation of all available methods."""
        md = "# RELDEC Method Catalog\n\n"
        md += f"*Generated: {self.timestamp}*\n\n"
        
        md += "## Overview\n\n"
        md += f"Total methods available: {len(METHOD_CATALOG)}\n\n"
        
        # Group methods by family
        families = {}
        for spec in METHOD_CATALOG:
            family = spec.family
            if family not in families:
                families[family] = []
            families[family].append(spec.name)
        
        md += "## Methods by Family\n\n"
        for family, methods in sorted(families.items()):
            md += f"### {family.replace('_', ' ').title()}\n"
            for method in sorted(methods):
                md += f"- `{method}`\n"
            md += "\n"
        
        return md

    def generate_training_policy_catalog_md(self) -> str:
        """Generate markdown documentation of all training policies."""
        md = "# Training Policy Catalog\n\n"
        md += f"*Generated: {self.timestamp}*\n\n"
        
        md += "## Available Policies\n\n"
        md += f"Total policies: {len(TRAINING_POLICY_CATALOG)}\n\n"
        
        # Group policies by base algorithm
        bases = {}
        for spec in TRAINING_POLICY_CATALOG:
            # Extract base name (before z parameter)
            base = spec.name.split('_z')[0] if '_z' in spec.name else spec.name
            if base not in bases:
                bases[base] = []
            bases[base].append(spec)
        
        for base, policies in sorted(bases.items()):
            md += f"### {base.replace('_', ' ').title()}\n"
            for policy in sorted(policies, key=lambda p: p.name):
                z = policy.parameters.get("z", "?")
                md += f"- `{policy.name}` (z={z})\n"
            md += "\n"
        
        return md

    def generate_run_history_md(self) -> str:
        """Generate markdown summary of recent runs."""
        md = "# Recent Experiment Runs\n\n"
        md += f"*Generated: {self.timestamp}*\n\n"
        
        # Training runs
        train_index = self.store.get_run_index("training")
        if train_index:
            md += "## Recent Training Runs\n\n"
            md += "| Run ID | Created | Policy | Status |\n"
            md += "|--------|---------|--------|--------|\n"
            for entry in train_index[:10]:  # Show latest 10
                md += f"| `{entry['run_id']}` | {entry['created_at'][:10]} | {entry['policy_type']} | ✓ |\n"
            md += "\n"
        
        # Evaluation runs
        eval_index = self.store.get_run_index("evaluation")
        if eval_index:
            md += "## Recent Evaluation Runs\n\n"
            md += "| Run ID | Created | Methods | Status |\n"
            md += "|--------|---------|---------|--------|\n"
            for entry in eval_index[:10]:  # Show latest 10
                methods = ", ".join(entry['methods'][:2])
                if len(entry['methods']) > 2:
                    methods += f", +{len(entry['methods']) - 2}"
                md += f"| `{entry['run_id']}` | {entry['created_at'][:10]} | {methods} | ✓ |\n"
            md += "\n"
        
        return md

    def generate_status_md(self) -> str:
        """Generate overall system status markdown."""
        md = "# RELDEC System Status\n\n"
        md += f"*Last updated: {self.timestamp}*\n\n"
        
        # Training runs
        train_runs = self.store.list_training_runs()
        eval_runs = self.store.list_evaluation_runs()
        
        md += "## Statistics\n\n"
        md += f"- Total training runs: {len(train_runs)}\n"
        md += f"- Total evaluation runs: {len(eval_runs)}\n"
        md += f"- Available methods: {len(METHOD_CATALOG)}\n"
        md += f"- Available policies: {len(TRAINING_POLICY_CATALOG)}\n"
        md += "\n"
        
        # Latest runs
        if train_runs:
            latest_train = train_runs[-1]
            md += f"- Latest training run: `{latest_train}`\n"
        if eval_runs:
            latest_eval = eval_runs[-1]
            md += f"- Latest evaluation run: `{latest_eval}`\n"
        
        md += "\n"
        return md

    def write_context_bundle(self, output_dir: str | Path = "docs") -> None:
        """Write all context markdown files to directory.
        
        Args:
            output_dir: Directory to write markdown files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write individual files
        files = {
            "METHODS.md": self.generate_method_catalog_md(),
            "POLICIES.md": self.generate_training_policy_catalog_md(),
            "RUNS.md": self.generate_run_history_md(),
            "STATUS.md": self.generate_status_md(),
        }
        
        for filename, content in files.items():
            filepath = output_dir / filename
            filepath.write_text(content, encoding="utf-8")
            print(f"[context-sync] wrote {filepath}")

    def generate_full_context(self) -> str:
        """Generate complete context as single markdown file.
        
        Returns:
            Complete markdown context
        """
        md = "# RELDEC Complete Context\n\n"
        md += f"*Generated: {self.timestamp}*\n\n"
        
        md += "---\n\n"
        md += "## System Status\n\n"
        md += self.generate_status_md()
        
        md += "---\n\n"
        md += "## Method Catalog\n\n"
        md += self.generate_method_catalog_md()
        
        md += "---\n\n"
        md += "## Training Policies\n\n"
        md += self.generate_training_policy_catalog_md()
        
        md += "---\n\n"
        md += "## Run History\n\n"
        md += self.generate_run_history_md()
        
        return md
