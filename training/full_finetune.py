"""
samvad/training/full_finetune.py

Full fine-tuning of Qwen2-0.5B-Instruct on IntentCONANv2.
All model weights are updated — no parameter freezing.

This is the baseline. All PEFT methods (LoRA, QLoRA, Prefix) are
compared against the results produced here.

Usage (via main.py):
    python main.py --train_full

Usage (directly):
    python training/full_finetune.py
"""

import logging
import os

import torch
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

# ── Disable HuggingFace network calls during training ─────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID = config.get("model.id")
PROCESSED_DIR = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR = config.get("paths.checkpoints_dir")
LEARNING_RATE = float(config.get("training.learning_rate"))
NUM_EPOCHS = int(config.get("training.num_epochs"))
BATCH_SIZE = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS = int(config.get("training.grad_accum_steps"))
LOGGING_STEPS = int(config.get("training.logging_steps"))
SAVE_STEPS = int(config.get("training.save_steps"))
LR_SCHEDULER = config.get("training.lr_scheduler")

RUN_NAME = "full-finetune"
OUTPUT_DIR = os.path.join(CHECKPOINTS_DIR, RUN_NAME)



def train(output_dir: str = None) -> None:
    """Run full fine-tuning.
    
    Args:
        output_dir: Optional override for output directory
    """
    final_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(final_output_dir, exist_ok=True)

    model = load_base_model(MODEL_ID, torch_dtype=torch.float32)
    tokenizer = load_tokenizer(MODEL_ID)
    train_dataset, val_dataset = load_dataset(PROCESSED_DIR)

    log_model_params(model)

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
        gradient_checkpointing=True,
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
    )

    last_checkpoint = find_last_checkpoint(final_output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving final model …")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log_training_complete(final_output_dir, metrics.get("train_loss"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    train()
