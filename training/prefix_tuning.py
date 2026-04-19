"""
samvad/training/prefix_tuning.py

Prefix Tuning of Qwen2-0.5B-Instruct on IntentCONANv2.

Unlike LoRA which modifies weight matrices, Prefix Tuning prepends learnable
virtual tokens to key and value tensors. Base model weights are completely
frozen — only prefix vectors (~10K params) are updated.

Most parameter-efficient method:
  Full FT       →  494M trainable params
  LoRA / QLoRA  →  ~3.4M trainable params
  Prefix Tuning →  ~10K  trainable params  (0.002%)

Usage (via main.py):
    python main.py --train --method prefix --model qwen

Usage (directly):
    python training/prefix_tuning.py
"""

import json
import logging
import os
import shutil

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    EarlyStoppingCallback,
)

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

MODEL_ID = config.get("model.id")
PROCESSED_DIR = config.processed_data_dir()
LEARNING_RATE = float(config.get("training.learning_rate"))
NUM_EPOCHS = int(config.get("training.num_epochs"))
BATCH_SIZE = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS = int(config.get("training.grad_accum_steps"))
LOGGING_STEPS = int(config.get("training.logging_steps"))
SAVE_STEPS = int(config.get("training.save_steps"))
LR_SCHEDULER = config.get("training.lr_scheduler")
NUM_VIRTUAL_TOKENS = int(config.get("prefix.num_virtual_tokens"))

RUN_NAME = "prefix-tuning"
OUTPUT_DIR = config.checkpoint_dir(RUN_NAME)



def load_model_and_tokenizer():
    """Load base model and wrap with Prefix Tuning config."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model(MODEL_ID, torch_dtype=torch.float32)

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prefix_projection=True,
    )
    model = get_peft_model(model, prefix_config)

    log_model_params(model)
    return model, tokenizer


def training_already_complete(checkpoint_path: str) -> bool:
    """Check if training finished from checkpoint state.
    
    Args:
        checkpoint_path: Path to a checkpoint directory
        
    Returns:
        True if training already reached NUM_EPOCHS, False otherwise
    """
    if checkpoint_path is None:
        return False
    
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_path):
        return False
    
    with open(state_path) as f:
        state = json.load(f)
    
    return state.get("epoch", 0) >= NUM_EPOCHS


def copy_best_checkpoint(best_checkpoint: str, final_output_dir: str) -> None:
    """Copy best checkpoint files to output directory.
    
    Args:
        best_checkpoint: Path to best checkpoint directory
        final_output_dir: Destination directory
    """
    if best_checkpoint and os.path.isdir(best_checkpoint):
        logger.info(f"Copying best checkpoint from: {best_checkpoint}")
        for f in os.listdir(best_checkpoint):
            src = os.path.join(best_checkpoint, f)
            dst = os.path.join(final_output_dir, f)
            shutil.copy2(src, dst)
    else:
        logger.info("No best checkpoint recorded — saving current model state.")


def train(output_dir: str = None) -> None:
    """Run Prefix Tuning fine-tuning.
    
    Args:
        output_dir: Optional override for output directory
    """
    final_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(final_output_dir, exist_ok=True)

    last_checkpoint = find_last_checkpoint(final_output_dir)
    if training_already_complete(last_checkpoint):
        logger.info(f"Training already completed at epoch {NUM_EPOCHS}. Skipping.")
        logger.info(f"Adapter already saved at: {final_output_dir}")
        return

    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    model, tokenizer = load_model_and_tokenizer()
    train_dataset, val_dataset = load_dataset(PROCESSED_DIR)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

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
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log_training_start(
        RUN_NAME,
        BATCH_SIZE * GRAD_ACCUM_STEPS,
        final_output_dir,
        virtual_tokens=NUM_VIRTUAL_TOKENS,
        prefix_projection="True",
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Manually copy best checkpoint since load_best_model_at_end=False
    copy_best_checkpoint(trainer.state.best_model_checkpoint, final_output_dir)
    
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
