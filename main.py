#!/usr/bin/env python3
"""
samvad/main.py

Main entry point for the samvad project.
Run the project with: python main.py
"""

import logging
import argparse
import os
from config import config
from data.preprocess import preprocess


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


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
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Starting samvad project")
    logger.info("=" * 60)
    
    # Log loaded configuration
    logger.info(f"Config loaded from: {config.config_path}")
    logger.info(f"Model ID: {config.get('model.id')}")
    logger.info(f"Dataset ID: {config.get('dataset.id')}")
    logger.info(f"Batch size: {config.get('training.batch_size')}")
    logger.info(f"Epochs: {config.get('training.num_epochs')}")
    
    # Create necessary directories
    for dir_key in ["processed_data_dir", "checkpoints_dir", "results_dir"]:
        dir_path = config.get(f"paths.{dir_key}")
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    logger.info("=" * 60)
    
    # Run preprocessing if requested
    if args.preprocess:
        logger.info("Running data preprocessing...")
        logger.info("=" * 60)
        preprocess(output_dir=args.output_dir, preview=args.preview)
    else:
        logger.info("Setup complete. Ready to run your code!")
        logger.info("=" * 60)
        logger.info("Tip: Use --preprocess to run data preprocessing")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
