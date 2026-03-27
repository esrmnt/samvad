"""
samvad/config module

Provides centralized configuration management accessible from anywhere in the project.

Usage:
    from config import config
    
    # Attribute access (for sections)
    model_id = config.model["id"]
    
    # Dot notation access
    batch_size = config.get("training.batch_size")
"""

from .config import Config, _config_instance

# Global config instance accessible throughout the project
config = _config_instance

__all__ = ["config", "Config"]