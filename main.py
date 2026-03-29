#!/usr/bin/env python3
"""
samvad/main.py

Main entry point for the samvad project.

Usage (from main entry point):

    python main.py                                      # just sets up directories and logs config
    python main.py --preprocess                         # saves to default path
    python main.py --preprocess --preview               # preview mode
    python main.py --preprocess --output_dir /path      # custom output dir
"""

import os
import logging
import argparse

from config import config
from data.preprocess import preprocess
from training.full_finetune import train as full_finetune

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def setup_directories() -> None:
    """Create necessary directories for data artifacts, checkpoints, and results."""
    artifacts_root = config.get("paths.data_artifacts_dir")
    if artifacts_root and not os.path.exists(artifacts_root):
        os.makedirs(artifacts_root, exist_ok=True)
    
    for dir_key in ["processed_data_dir", "checkpoints_dir", "results_dir"]:
        dir_path = config.get(f"paths.{dir_key}")
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

def main() -> None:
    """Main entry point for the samvad project."""
    parser = argparse.ArgumentParser(description="samvad: Counterspeech generation")
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run data preprocessing"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory for processed data (used with --preprocess)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview sample prompts without saving (used with --preprocess)"
    )
    parser.add_argument(
        "--train_full", action="store_true",
        help="Run full fine-tuning (training/full_finetune.py)"
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting samvad project")
    
    # Log loaded configuration
    logger.info(f"Config loaded from: {config.config_path}")
    logger.info(f"Model ID: {config.get('model.id')}")
    logger.info(f"Dataset ID: {config.get('dataset.id')}")
    logger.info(f"Batch size: {config.get('training.batch_size')}")
    logger.info(f"Epochs: {config.get('training.num_epochs')}")
    
    # Create main dataset directory and subdirectories
    setup_directories()
    
    # Run preprocessing if requested
    if args.preprocess:
        logger.info("Running data preprocessing...")
        preprocess(output_dir=args.output_dir, preview=args.preview)

    # Run full fine-tuning if requested
    if args.train_full:
        logger.info("Running full fine-tuning...")
        full_finetune()
        return

    if not args.preprocess and not args.train_full:
        logger.info("Setup complete. Ready to run your code!")
        logger.info("Tip: Use --preprocess to run data preprocessing")
    
if __name__ == "__main__":
    main()
