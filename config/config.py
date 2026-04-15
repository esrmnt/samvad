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