"""File-first persistence store for RELDEC runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .experiments import RunManifest, EvaluationManifest


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_config_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def compute_config_hash(config: Dict[str, Any]) -> str:
    normalized = _normalize_config_value(config)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunStore:
    """Persistent storage for training and evaluation runs."""

    def __init__(self, base_dir: str | Path = "runs"):
        """Initialize the run store.
        
        Args:
            base_dir: Root directory for storing runs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.base_dir / "training"
        self.eval_dir = self.base_dir / "evaluation"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def save_training_run(self, manifest: RunManifest, artifacts_dir: str | Path) -> Path:
        """Save a training run manifest and link to artifacts.
        
        Args:
            manifest: RunManifest instance
            artifacts_dir: Directory containing training artifacts (checkpoints, logs, etc.)
            
        Returns:
            Path to saved manifest JSON
        """
        run_dir = self.train_dir / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        
        # Create symlink to artifacts directory for easy access
        artifacts_link = run_dir / "artifacts"
        if artifacts_link.is_symlink() or artifacts_link.exists():
            artifacts_link.unlink()
        artifacts_link.symlink_to(Path(artifacts_dir).resolve(), target_is_directory=True)
        
        return manifest_path

    def save_evaluation_run(self, manifest: EvaluationManifest, artifacts_dir: str | Path) -> Path:
        """Save an evaluation run manifest and link to artifacts.
        
        Args:
            manifest: EvaluationManifest instance
            artifacts_dir: Directory containing evaluation artifacts (results CSV, etc.)
            
        Returns:
            Path to saved manifest JSON
        """
        run_dir = self.eval_dir / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        
        # Create symlink to artifacts directory for easy access
        artifacts_link = run_dir / "artifacts"
        if artifacts_link.is_symlink() or artifacts_link.exists():
            artifacts_link.unlink()
        artifacts_link.symlink_to(Path(artifacts_dir).resolve(), target_is_directory=True)
        
        return manifest_path

    def list_training_runs(self) -> List[str]:
        """List all training run IDs."""
        return sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])

    def list_evaluation_runs(self) -> List[str]:
        """List all evaluation run IDs."""
        return sorted([d.name for d in self.eval_dir.iterdir() if d.is_dir()])

    def load_training_manifest(self, run_id: str) -> Optional[RunManifest]:
        """Load a training run manifest by ID.
        
        Args:
            run_id: Training run ID
            
        Returns:
            RunManifest instance or None if not found
        """
        manifest_path = self.train_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            return None
        
        manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Reconstruct RunManifest from dict (simplified version)
        # In a real implementation, you'd have proper deserialization
        return RunManifest(
            run_id=manifest_dict["run_id"],
            created_at_utc=manifest_dict["created_at_utc"],
            experiment=manifest_dict["experiment"],
            training_config=manifest_dict["training_config"],
            artifacts=manifest_dict["artifacts"],
        )

    def load_evaluation_manifest(self, run_id: str) -> Optional[EvaluationManifest]:
        """Load an evaluation run manifest by ID.
        
        Args:
            run_id: Evaluation run ID
            
        Returns:
            EvaluationManifest instance or None if not found
        """
        manifest_path = self.eval_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            return None
        
        manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Simplified version - in practice, deserialize properly
        return EvaluationManifest(
            run_id=manifest_dict["run_id"],
            created_at_utc=manifest_dict["created_at_utc"],
            experiment=manifest_dict["experiment"],
            evaluation_config=manifest_dict["evaluation_config"],
            artifacts=manifest_dict["artifacts"],
        )

    def get_run_index(self, run_type: str = "training") -> List[Dict[str, Any]]:
        """Get an index of all runs with metadata.
        
        Args:
            run_type: "training" or "evaluation"
            
        Returns:
            List of dicts with run metadata (ID, policy, methods, date, etc.)
        """
        if run_type == "training":
            run_ids = self.list_training_runs()
            manifests = [self.load_training_manifest(rid) for rid in run_ids]
        elif run_type == "evaluation":
            run_ids = self.list_evaluation_runs()
            manifests = [self.load_evaluation_manifest(rid) for rid in run_ids]
        else:
            raise ValueError(f"Unknown run_type: {run_type}")
        
        index = []
        for manifest in manifests:
            if manifest is None:
                continue
            
            if run_type == "training":
                entry = {
                    "run_id": manifest.run_id,
                    "created_at": manifest.created_at_utc,
                    "policy_type": manifest.experiment.get("policy_type"),
                    "code": manifest.experiment.get("code"),
                    "matrix_csv": manifest.experiment.get("matrix_csv"),
                }
            else:  # evaluation
                entry = {
                    "run_id": manifest.run_id,
                    "created_at": manifest.created_at_utc,
                    "methods": manifest.experiment.get("methods", []),
                    "code": manifest.experiment.get("code"),
                    "matrix_csv": manifest.experiment.get("matrix_csv"),
                }
            
            index.append(entry)
        
        return sorted(index, key=lambda x: x["created_at"], reverse=True)

    def generate_run_summary_markdown(self, run_type: str = "training") -> str:
        """Generate a markdown summary of all runs.
        
        Args:
            run_type: "training" or "evaluation"
            
        Returns:
            Markdown-formatted run summary
        """
        index = self.get_run_index(run_type)
        
        if run_type == "training":
            md = "# Training Runs\n\n"
            md += "| Run ID | Created | Policy | Code | Status |\n"
            md += "|--------|---------|--------|------|--------|\n"
            for entry in index:
                status = "✓ Complete"
                md += f"| {entry['run_id']} | {entry['created_at'][:10]} | {entry['policy_type']} | {entry['code']} | {status} |\n"
        else:  # evaluation
            md = "# Evaluation Runs\n\n"
            md += "| Run ID | Created | Methods | Code | Status |\n"
            md += "|--------|---------|---------|------|--------|\n"
            for entry in index:
                methods_str = ", ".join(entry['methods'][:3])
                if len(entry['methods']) > 3:
                    methods_str += f", +{len(entry['methods']) - 3}"
                status = "✓ Complete"
                md += f"| {entry['run_id']} | {entry['created_at'][:10]} | {methods_str} | {entry['code']} | {status} |\n"
        
        return md
