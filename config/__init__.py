"""
samvad/config module

Provides centralized configuration management accessible from anywhere in the project.

Usage:
    from config import config
    
    # Dot notation access
    batch_size = config.get("training.batch_size")
    model_id = config.get("model.id")
"""

from .config import Config, config

__all__ = ["config", "Config"]
