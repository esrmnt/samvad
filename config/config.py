"""
samvad/config/config.py

Configuration loader that reads from config.yaml and provides structured access.
Use: from config import config
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration class that loads settings from YAML and provides attribute access."""

    _instance: Optional["Config"] = None

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        """Implement singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize config by loading from YAML file.
        
        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if self._initialized:
            return
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        self.config_path = Path(config_path)
        self._config_dict: Dict[str, Any] = {}
        self._load_yaml()
        self._initialized = True

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
            self._config_dict = data if data else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., 'model.id', 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
            
        Examples:
            config.get("model.id")
            config.get("training.batch_size", 8)
        """
        keys = key.split(".")
        value = self._config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def model_names(self) -> list:
        """Return configured model aliases."""
        models = self.get("models", {})
        return sorted(models.keys()) if isinstance(models, dict) else []

    def select_model(self, model_name: Optional[str]) -> None:
        """Select the active model config by alias."""
        if not model_name:
            return

        models = self.get("models", {})
        if not isinstance(models, dict) or model_name not in models:
            choices = ", ".join(self.model_names()) or "none configured"
            raise ValueError(f"Unknown model '{model_name}'. Available models: {choices}")

        selected = models[model_name]
        model_config = self._config_dict.setdefault("model", {})
        model_config["name"] = model_name
        model_config["id"] = selected["id"]
        if selected.get("slug"):
            model_config["slug"] = selected["slug"]
        else:
            model_config.pop("slug", None)

    def model_slug(self) -> str:
        """Return the folder-safe model identifier used for artifacts."""
        configured_slug = self.get("model.slug")
        if configured_slug:
            return configured_slug

        model_id = self.get("model.id", "model")
        slug = model_id.rsplit("/", 1)[-1].lower()
        slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "model"

    def artifact_path(self, *parts: str) -> str:
        """Build a path under the configured artifact root."""
        return os.path.join(self.get("paths.data_artifacts_dir", "./dataset"), *parts)

    def processed_data_dir(self) -> str:
        """Return the model-scoped processed dataset directory."""
        return os.path.join(self.get("paths.processed_data_dir"), self.model_slug())

    def checkpoints_dir(self) -> str:
        """Return the model-scoped checkpoint root directory."""
        return os.path.join(self.get("paths.checkpoints_dir"), self.model_slug())

    def checkpoint_dir(self, run_name: str) -> str:
        """Return the checkpoint directory for one training run."""
        return os.path.join(self.checkpoints_dir(), run_name)

    def results_dir(self) -> str:
        """Return the model-scoped results root directory."""
        return os.path.join(self.get("paths.results_dir"), self.model_slug())

    def generated_dir(self) -> str:
        """Return the directory for generated prediction CSVs."""
        return os.path.join(self.results_dir(), "generated")

    def metrics_dir(self) -> str:
        """Return the directory for evaluation metric outputs."""
        return os.path.join(self.results_dir(), "metrics")

    def artifact_dirs(self) -> list:
        """Return artifact directories that should exist before running tasks."""
        return [
            self.get("paths.data_artifacts_dir"),
            self.get("paths.processed_data_dir"),
            self.get("paths.checkpoints_dir"),
            self.get("paths.results_dir"),
            self.processed_data_dir(),
            self.checkpoints_dir(),
            self.results_dir(),
            self.generated_dir(),
            self.metrics_dir(),
        ]

    def __getattr__(self, item: str) -> Any:
        """Allow attribute-style access to config values."""
        if item.startswith("_"):
            raise AttributeError(f"Config attribute '{item}' not found")
        value = self._config_dict.get(item)
        if value is None:
            raise AttributeError(f"Config key '{item}' not found")
        return value

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Load configuration on import
config = Config()
