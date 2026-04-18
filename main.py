#!/usr/bin/env python3
"""
samvad/main.py

Main entry point for the samvad counterspeech generation project.

This module orchestrates data preprocessing and model fine-tuning workflows.
All configuration is loaded from config/config.yaml.

Usage:
    python main.py                                  # Setup only
    python main.py --preprocess                     # Preprocess data
    python main.py --preprocess --preview           # Preview prompts
    python main.py --train_full                     # Full fine-tuning
    python main.py --train_lora                     # LoRA fine-tuning
    python main.py --train_qlora                    # QLoRA fine-tuning
    python main.py --train_prefix                   # Prefix tuning
"""

import os
import logging
import argparse
from typing import Callable, Dict

from config import config

from loaders.preprocess import preprocess

from training.lora import train as train_lora
from training.qlora import train as train_qlora
from training.prefix_tuning import train as train_prefix
from training.full_finetune import train as train_full_finetune

from evaluation.generate import generate
from evaluation.evaluate import evaluate


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def setup_directories() -> None:
    """Create necessary directories for artifacts, checkpoints, and results."""
    required_dirs = [
        "paths.data_artifacts_dir",
        "paths.processed_data_dir",
        "paths.checkpoints_dir",
        "paths.results_dir",
    ]
    
    for dir_key in required_dirs:
        dir_path = config.get(dir_key)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


def log_config() -> None:
    """Log loaded configuration to console."""
    logger = logging.getLogger(__name__)
    logger.info(f"Config loaded from: {config.config_path}")
    logger.info(f"Model ID: {config.get('model.id')}")
    logger.info(f"Dataset ID: {config.get('dataset.id')}")
    logger.info(f"Batch size: {config.get('training.batch_size')}")
    logger.info(f"Epochs: {config.get('training.num_epochs')}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="samvad: Counterspeech generation with fine-tuned language models"
    )
    
    # Preprocessing arguments
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing pipeline")
    parser.add_argument("--preview", action="store_true", help="Preview sample prompts without saving (used with --preprocess)")
    parser.add_argument("--output_dir", default=None, help="Output directory for processed data (used with --preprocess)")
    
    # Training arguments
    parser.add_argument("--train_full", action="store_true", help="Run full fine-tuning")
    parser.add_argument("--train_lora", action="store_true", help="Run LoRA fine-tuning")
    parser.add_argument("--train_qlora", action="store_true", help="Run QLoRA fine-tuning")
    parser.add_argument("--train_prefix", action="store_true", help="Run prefix tuning fine-tuning")

    parser.add_argument("--generate",  action="store_true")
    parser.add_argument("--evaluate",  action="store_true")
    
    return parser


def main() -> None:
    """Main entry point for the samvad project."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting samvad project")
    log_config()
    setup_directories()
    
    # Map training arguments to their corresponding functions
    training_tasks: Dict[str, Callable] = {
        "preprocess": lambda: preprocess(output_dir=args.output_dir, preview=args.preview),
        "train_full": train_full_finetune,
        "train_lora": train_lora,
        "train_qlora": train_qlora,
        "train_prefix": train_prefix,
        "generate": generate,
        "evaluate": evaluate,
    }
    
    # Execute requested task
    any_task_executed = False
    for arg_name, task_func in training_tasks.items():
        if getattr(args, arg_name):
            logger.info(f"Running {arg_name.replace('_', ' ')}...")
            task_func()
            any_task_executed = True
            # Training tasks exit after execution
            if arg_name != "preprocess":
                return
    
    if not any_task_executed:
        logger.info("Setup complete. Ready to run your code!")
        logger.info("Use --preprocess to run data preprocessing")
        logger.info("Use --train_full, --train_lora, --train_qlora, or --train_prefix to train")


if __name__ == "__main__":
    main()
