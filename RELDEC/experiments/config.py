"""Configuration file support for RELDEC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

try:
    import yaml
except ImportError:
    yaml = None  # Optional dependency

from .spec import ExperimentSpec, EvaluationSpec


class ConfigLoader:
    """Load and validate RELDEC experiment configurations."""

    @staticmethod
    def load_yaml(config_path: str | Path) -> Dict[str, Any]:
        """Load YAML configuration file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            Dictionary of configuration
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file not found
        """
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files. Install with: pip install pyyaml")
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config or {}

    @staticmethod
    def load_json(config_path: str | Path) -> Dict[str, Any]:
        """Load JSON configuration file.
        
        Args:
            config_path: Path to JSON config file
            
        Returns:
            Dictionary of configuration
            
        Raises:
            FileNotFoundError: If config file not found
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        return config

    @staticmethod
    def load(config_path: str | Path) -> Dict[str, Any]:
        """Load configuration file (auto-detect format).
        
        Args:
            config_path: Path to config file (JSON or YAML)
            
        Returns:
            Dictionary of configuration
        """
        config_path = Path(config_path)
        
        if config_path.suffix in {".yaml", ".yml"}:
            return ConfigLoader.load_yaml(config_path)
        elif config_path.suffix == ".json":
            return ConfigLoader.load_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    @staticmethod
    def training_config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert training config dict to command-line argument equivalents.
        
        This makes it easy to merge config files with CLI args.
        
        Args:
            config: Configuration dictionary (from YAML/JSON)
            
        Returns:
            Dictionary with keys matching argument parser dest names
        """
        args_dict = {}
        
        # Top-level experiment settings
        if "experiment" in config:
            exp = config["experiment"]
            args_dict.setdefault("code", exp.get("code", "ab"))
            args_dict.setdefault("matrix_csv", exp.get("matrix_csv"))
        
        # Training settings
        if "training" in config:
            train = config["training"]
            args_dict.setdefault("snr_db", train.get("snr_db"))
            args_dict.setdefault("episodes_per_snr", train.get("episodes_per_snr", 2500))
            args_dict.setdefault("policy_type", train.get("policy_type"))
        
        # Hyperparameters
        if "hyperparams" in config:
            hp = config["hyperparams"]
            args_dict.setdefault("alpha", hp.get("alpha", 0.1))
            args_dict.setdefault("beta", hp.get("beta", 0.9))
            args_dict.setdefault("epsilon", hp.get("epsilon", 0.6))
            args_dict.setdefault("l_max", hp.get("l_max", 50))
        
        # Method parameters
        if "parameters" in config:
            params = config["parameters"]
            args_dict.setdefault("z", params.get("z"))
            args_dict.setdefault("mi_bins", params.get("mi_bins", 21))
        
        # Checkpoint settings
        if "checkpoint" in config:
            ckpt = config["checkpoint"]
            args_dict.setdefault("checkpoint_every_episodes", ckpt.get("every_episodes", 100))
            args_dict.setdefault("no_history", ckpt.get("no_history", False))
        
        # System settings
        if "system" in config:
            sys_cfg = config["system"]
            args_dict.setdefault("device", sys_cfg.get("device", "cpu"))
            args_dict.setdefault("seed", sys_cfg.get("seed"))
        
        return args_dict

    @staticmethod
    def evaluation_config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert evaluation config dict to command-line argument equivalents.
        
        Args:
            config: Configuration dictionary (from YAML/JSON)
            
        Returns:
            Dictionary with keys matching argument parser dest names
        """
        args_dict = {}
        
        # Top-level experiment settings
        if "experiment" in config:
            exp = config["experiment"]
            args_dict.setdefault("code", exp.get("code", "ab"))
            args_dict.setdefault("matrix_csv", exp.get("matrix_csv"))
        
        # Evaluation settings
        if "evaluation" in config:
            ev = config["evaluation"]
            args_dict.setdefault("methods", ev.get("methods", []))
            args_dict.setdefault("snr_db", ev.get("snr_db"))
            args_dict.setdefault("i_max", ev.get("i_max", 100))
            args_dict.setdefault("target_frame_errors", ev.get("target_frame_errors", 100))
            args_dict.setdefault("max_frames", ev.get("max_frames", 100000))
        
        # Method parameters
        if "parameters" in config:
            params = config["parameters"]
            args_dict.setdefault("z", params.get("z"))
            args_dict.setdefault("mi_bins", params.get("mi_bins", 21))
        
        # Checkpoint paths
        if "checkpoints" in config:
            ckpts = config["checkpoints"]
            args_dict.setdefault("q_table", ckpts.get("q_table"))
            args_dict.setdefault("mi_tabular_q_table", ckpts.get("mi_tabular_q_table"))
            args_dict.setdefault("deep_checkpoint", ckpts.get("deep_checkpoint"))
        
        # System settings
        if "system" in config:
            sys_cfg = config["system"]
            args_dict.setdefault("seed", sys_cfg.get("seed", 42))
            args_dict.setdefault("random_codewords", sys_cfg.get("random_codewords", False))
        
        return args_dict
