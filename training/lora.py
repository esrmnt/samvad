"""
samvad/training/lora.py

LoRA fine-tuning of Qwen2-0.5B-Instruct on IntentCONANv2.
Only the low-rank adapter matrices A and B are updated (~0.7% of params).
Base model weights are completely frozen.

Usage (via main.py):
    python main.py --train --method lora --model qwen

Usage (directly):
    python training/lora.py
"""

import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, EarlyStoppingCallback

from config import config
from .common import (
    load_base_model,
    load_tokenizer,
    load_dataset,
    log_model_params,
    build_common_training_args,
    find_last_checkpoint,
    log_training_start,
    log_training_complete,
)

logger = logging.getLogger(__name__)

os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = config.get("model.id")
PROCESSED_DIR = config.processed_data_dir()
LEARNING_RATE = float(config.get("training.learning_rate"))
NUM_EPOCHS = int(config.get("training.num_epochs"))
BATCH_SIZE = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS = int(config.get("training.grad_accum_steps"))
LOGGING_STEPS = int(config.get("training.logging_steps"))
SAVE_STEPS = int(config.get("training.save_steps"))
LR_SCHEDULER = config.get("training.lr_scheduler")

LORA_R = int(config.get("lora.r"))
LORA_ALPHA = int(config.get("lora.alpha"))
LORA_DROPOUT = float(config.get("lora.dropout"))
LORA_TARGETS = config.get("lora.target_modules")

RUN_NAME = "lora"
OUTPUT_DIR = config.checkpoint_dir(RUN_NAME)



def load_model_and_tokenizer():
    """Load base model in float16 and wrap with LoRA adapters."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model(MODEL_ID, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    log_model_params(model)
    return model, tokenizer


def train(output_dir: str = None) -> None:
    """Run LoRA fine-tuning.
    
    Args:
        output_dir: Optional override for output directory
    """
    final_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(final_output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = load_dataset(PROCESSED_DIR)

    training_args = build_common_training_args(
        output_dir=final_output_dir,
        run_name=RUN_NAME,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        lr_scheduler=LR_SCHEDULER,
        fp16=True,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log_training_start(
        RUN_NAME,
        BATCH_SIZE * GRAD_ACCUM_STEPS,
        final_output_dir,
        rank_r=LORA_R,
        alpha=LORA_ALPHA,
        target_modules=", ".join(LORA_TARGETS),
    )

    last_checkpoint = find_last_checkpoint(final_output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving LoRA adapter …")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log_training_complete(final_output_dir, metrics.get("train_loss"))
    logger.info(f"Train loss       : {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    train()
