"""
samvad/training/common.py

Shared utilities for all fine-tuning methods.
Reduces duplication across full_finetune, lora, qlora, and prefix_tuning modules.
"""

import os
import logging
from typing import Tuple

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def get_tensor_columns() -> set:
    """Return the set of tensor column names used in training."""
    return {"input_ids", "attention_mask", "labels"}


def load_dataset(processed_dir: str) -> Tuple:
    """Load preprocessed Arrow dataset and keep only tensor columns.
    
    Args:
        processed_dir: Path to preprocessed dataset directory
        
    Returns:
        Tuple of (train_dataset, val_dataset) with only tensor columns
        
    Raises:
        FileNotFoundError: If processed data directory doesn't exist
    """
    logger.info(f"Loading dataset from: {processed_dir}")

    if not os.path.exists(processed_dir):
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}. "
            "Run: python main.py --preprocess"
        )

    dataset = load_from_disk(processed_dir)
    tensor_cols = get_tensor_columns()

    train_dataset = dataset["train"].remove_columns(
        [c for c in dataset["train"].column_names if c not in tensor_cols]
    )
    val_dataset = dataset["val"].remove_columns(
        [c for c in dataset["val"].column_names if c not in tensor_cols]
    )
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    logger.info(f"Train : {len(train_dataset):,} samples")
    logger.info(f"Val   : {len(val_dataset):,} samples")
    
    return train_dataset, val_dataset


def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load and configure tokenizer.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Configured tokenizer with pad token and right padding
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_id: str, torch_dtype: torch.dtype = torch.float32) -> AutoModelForCausalLM:
    """Load base model with common configuration.
    
    Args:
        model_id: HuggingFace model identifier
        torch_dtype: PyTorch dtype for the model
        
    Returns:
        Loaded model with standard config settings
    """
    logger.info(f"Loading base model: {model_id}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model


def log_model_params(model: AutoModelForCausalLM) -> None:
    """Log total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")


def find_last_checkpoint(checkpoint_dir: str) -> str:
    """Find the most recent checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to most recent checkpoint, or None if none found
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    
    checkpoints = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ]
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None


def build_common_training_args(
    output_dir: str,
    run_name: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    grad_accum_steps: int,
    logging_steps: int,
    save_steps: int,
    lr_scheduler: str,
    fp16: bool = True,
    bf16: bool = False,
    optim: str = "adamw_torch_fused",
    **kwargs
) -> TrainingArguments:
    """Build TrainingArguments with sensible defaults for all fine-tuning methods.
    
    Args:
        output_dir: Output directory for checkpoints
        run_name: Name for this training run
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Per-device batch size
        grad_accum_steps: Gradient accumulation steps
        logging_steps: Logging frequency
        save_steps: Checkpoint save frequency
        lr_scheduler: Learning rate scheduler type
        fp16: Use float16 precision
        bf16: Use bfloat16 precision
        optim: Optimizer type
        **kwargs: Additional arguments to pass to TrainingArguments
        
    Returns:
        Configured TrainingArguments instance
    """
    default_args = {
        "output_dir": output_dir,
        "run_name": run_name,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "report_to": "none",
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler_type": lr_scheduler,
        "warmup_steps": 100,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "fp16": fp16,
        "bf16": bf16,
        "optim": optim,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "eval_strategy": "steps",
        "eval_steps": save_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": 42,
        "data_seed": 42,
        "remove_unused_columns": False,
    }
    
    # Override with any custom kwargs
    default_args.update(kwargs)
    
    return TrainingArguments(**default_args)


def log_training_start(run_name: str, effective_batch_size: int, checkpoint_dir: str, **info) -> None:
    """Log standardized training start message with configuration details.
    
    Args:
        run_name: Name of the training run
        effective_batch_size: Actual batch size (batch_size * grad_accum_steps)
        checkpoint_dir: Directory where checkpoints will be saved
        **info: Additional key-value pairs to log
    """
    logger.info(f"Starting {run_name} …")
    logger.info(f"Effective batch : {effective_batch_size}")
    logger.info(f"Checkpoint dir  : {checkpoint_dir}")
    for key, value in info.items():
        logger.info(f"{key.replace('_', ' ').title():<15}: {value}")


def log_training_complete(output_dir: str, train_loss: float = None) -> None:
    """Log standardized training completion message.
    
    Args:
        output_dir: Directory where model was saved
        train_loss: Final training loss
    """
    logger.info("Training complete.")
    logger.info(f"Model saved to : {output_dir}")
    if train_loss is not None:
        logger.info(f"Train loss     : {train_loss:.4f}")
