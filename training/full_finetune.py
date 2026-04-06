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
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from config import config

logger = logging.getLogger(__name__)

# ── Disable HuggingFace network calls during training ─────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID          = config.get("model.id")
PROCESSED_DIR     = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR   = config.get("paths.checkpoints_dir")
LEARNING_RATE     = float(config.get("training.learning_rate"))
NUM_EPOCHS        = int(config.get("training.num_epochs"))
BATCH_SIZE        = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS  = int(config.get("training.grad_accum_steps"))
WARMUP_STEPS      = 100
LOGGING_STEPS     = int(config.get("training.logging_steps"))
SAVE_STEPS        = int(config.get("training.save_steps"))
LR_SCHEDULER      = config.get("training.lr_scheduler")

RUN_NAME          = "full-finetune"
OUTPUT_DIR        = os.path.join(CHECKPOINTS_DIR, RUN_NAME)

def load_model_and_tokenizer():
    """
    Load base model in float32 and tokenizer.
    Trainer handles fp16 casting via TrainingArguments.
    """
    logger.info(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    return model, tokenizer

def load_dataset():
    """Load preprocessed Arrow dataset and strip non-tensor columns."""
    logger.info(f"Loading dataset from: {PROCESSED_DIR}")

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DIR}. "
            "Run: python main.py --preprocess"
        )

    dataset     = load_from_disk(PROCESSED_DIR)
    tensor_cols = {"input_ids", "attention_mask", "labels"}

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

def build_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        run_name                    = RUN_NAME,
        logging_strategy            = "steps",
        logging_steps               = LOGGING_STEPS,
        report_to                   = "none",
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_steps                = WARMUP_STEPS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        fp16                        = True,
        optim                       = "adamw_torch_fused",
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        seed                        = 42,
        data_seed                   = 42,
        remove_unused_columns       = False,
    )

def train(output_dir: str = None) -> None:
    """Run full fine-tuning."""
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer           = load_model_and_tokenizer()
    train_dataset, val_dataset = load_dataset()

    # No custom collator needed — data is already padded to MAX_LENGTH
    # in preprocessing. Default collator simply stacks tensors.
    trainer = Trainer(
        model            = model,
        args             = build_training_args(),
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = tokenizer,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting full fine-tuning …")
    logger.info(f"Effective batch size : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"Checkpoint dir       : {OUTPUT_DIR}")

    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [
            os.path.join(OUTPUT_DIR, d)
            for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving final model …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training complete.")
    logger.info(f"Model saved to : {OUTPUT_DIR}")
    logger.info(f"Train loss     : {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    train()
