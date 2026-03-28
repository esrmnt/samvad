"""
samvad/config/config.py

Configuration loader that reads from config.yaml and provides structured access.
Use: from config import config
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration class that loads settings from YAML and provides attribute access."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config by loading from YAML file.
        
        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        self.config_path = Path(config_path)
        self._config_dict: Dict[str, Any] = {}
        self._load_yaml()

    def _load_yaml(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)
            if data:
                self._config_dict = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation.
        
        Examples:
            config.get("model.id")  # returns model ID
            config.get("training.batch_size")  # returns batch size
        """
        keys = key.split(".")
        value = self._config_dict
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def __getattr__(self, item: str) -> Any:
        """Allow attribute-style access to config values."""
        if item.startswith("_"):
            return object.__getattribute__(self, item)
        value = self._config_dict.get(item)
        if value is None:
            raise AttributeError(f"Config key '{item}' not found")
        return value

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Load configuration on import
_config_instance = Config()