#!/usr/bin/env python3
"""
samvad/main.py

Main entry point for the samvad counterspeech generation project.

This module orchestrates data preprocessing and model fine-tuning workflows.
All configuration is loaded from config/config.yaml.

Usage:
    python main.py                                                          # Setup only
    python main.py --preprocess --model qwen                                # Preprocess data
    python main.py --train --method qlora --model qwen                      # Train model with QLoRA 
    python main.py --generate --model qwen                                  # Generate predictions
    python main.py --evaluate --model qwen                                  # Evaluate model performance
"""

import os
import logging
import argparse
from typing import Callable, Dict, Optional

from config import config


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
    for dir_path in config.artifact_dirs():
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)


def log_config() -> None:
    """Log loaded configuration to console."""
    logger = logging.getLogger(__name__)
    logger.info(f"Config loaded from: {config.config_path}")
    logger.info(f"Model name: {config.get('model.name', 'custom')}")
    logger.info(f"Model ID: {config.get('model.id')}")
    logger.info(f"Model slug: {config.model_slug()}")
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
    model_names = config.model_names()

    # Preprocessing arguments
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing pipeline")
    parser.add_argument("--preview", action="store_true", help="Preview sample prompts without saving (used with --preprocess)")
    parser.add_argument("--output_dir", default=None, help="Output directory for processed data (used with --preprocess)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--method",
        choices=["full", "full_finetune", "lora", "qlora", "prefix", "prefix_tuning"],
        help="Training method to use with --train",
    )
    parser.add_argument("--model",
        default=config.get("model.name"),
        choices=model_names or None,
        help="Model alias from config/config.yaml",
    )
    

    parser.add_argument("--generate", action="store_true", help="Generate predictions using trained model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance on test set")
    
    return parser


def normalize_method(method: Optional[str]) -> Optional[str]:
    """Normalize CLI method aliases to internal training keys."""
    method_aliases = {
        "full": "full_finetune",
        "lora": "lora",
        "qlora": "qlora",
        "prefix": "prefix_tuning"
    }
    return method_aliases.get(method) if method else None


def build_tasks(args: argparse.Namespace) -> Dict[str, Callable]:
    """Import task modules after model selection and return runnable tasks."""
    tasks: Dict[str, Callable] = {}
    if args.preprocess:
        from loaders.preprocess import preprocess

        tasks["preprocess"] = lambda: preprocess(output_dir=args.output_dir, preview=args.preview)

    method = normalize_method(args.method)
    if args.train:
        tasks[f"train_{method}"] = load_training_task(method)

    if args.generate:
        from evaluation.generate import generate

        tasks["generate"] = generate
    if args.evaluate:
        from evaluation.evaluate import evaluate

        tasks["evaluate"] = evaluate

    return tasks


def load_training_task(method: str) -> Callable:
    """Import and return the selected training function."""
    if method == "full_finetune":
        from training.full_finetune import train
    elif method == "lora":
        from training.lora import train
    elif method == "qlora":
        from training.qlora import train
    elif method == "prefix_tuning":
        from training.prefix_tuning import train
    else:
        raise ValueError(f"Unsupported training method: {method}")
    return train


def main() -> None:
    """Main entry point for the samvad project."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate argument dependencies
    if args.preview and not args.preprocess:
        parser.error("--preview requires --preprocess to be set")
    if args.train and not args.method:
        parser.error("--train requires --method")
    if args.method and not args.train:
        parser.error("--method requires --train")

    try:
        config.select_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting samvad project")
    log_config()
    setup_directories()
    
    tasks = build_tasks(args)
    
    # Execute requested task
    for task_name, task_func in tasks.items():
        logger.info(f"Running {task_name.replace('_', ' ')}...")
        task_func()
        if task_name != "preprocess":
            return
    
    if not tasks:
        logger.info("Setup complete. Ready to run your code!")
        logger.info("Use --preprocess --model qwen to run data preprocessing")
        logger.info("Use --train --method qlora --model qwen to train")
        logger.info("Use --generate --model qwen to generate predictions")
        logger.info("Use --evaluate --model qwen to evaluate outputs")


if __name__ == "__main__":
    main()
