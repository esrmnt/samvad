"""
samvad/training/prefix_tuning.py

Prefix Tuning of Qwen2-0.5B-Instruct on IntentCONANv2.

Unlike LoRA which modifies weight matrices, Prefix Tuning prepends a set
of learnable virtual tokens to the key and value tensors at every attention
layer. The base model weights are completely frozen — only the prefix
vectors (~10K params) are updated.

This is the most parameter-efficient method in the project:
  Full FT       →  494M trainable params
  LoRA / QLoRA  →  ~3.4M trainable params
  Prefix Tuning →  ~10K  trainable params  (0.002%)

Expected outcome: fastest training, lowest capacity, likely lower scores
than LoRA — this tradeoff is the interesting finding for your report.

Usage (via main.py):
    python main.py --train_prefix

Usage (directly):
    python training/prefix_tuning.py
"""

import logging
import os

import torch
from datasets import load_from_disk
from peft import (
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import config

logger = logging.getLogger(__name__)

MODEL_ID             = config.get("model.id")
PROCESSED_DIR        = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR      = config.get("paths.checkpoints_dir")
LEARNING_RATE        = float(config.get("training.learning_rate"))
NUM_EPOCHS           = int(config.get("training.num_epochs"))
BATCH_SIZE           = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS     = int(config.get("training.grad_accum_steps"))
WARMUP_STEPS         = 100
LOGGING_STEPS        = int(config.get("training.logging_steps"))
SAVE_STEPS           = int(config.get("training.save_steps"))
LR_SCHEDULER         = config.get("training.lr_scheduler")
NUM_VIRTUAL_TOKENS   = int(config.get("prefix.num_virtual_tokens"))

RUN_NAME             = "prefix-tuning"
OUTPUT_DIR           = os.path.join(CHECKPOINTS_DIR, RUN_NAME)


def load_model_and_tokenizer():
    """
    Load base model and wrap with Prefix Tuning config.

    PrefixTuningConfig creates a small MLP (called the prefix encoder)
    that maps a set of virtual token indices to prefix vectors.
    These vectors are prepended to the K and V tensors at every
    attention layer during the forward pass.

    The base model weights are frozen automatically by get_peft_model().
    Only the prefix encoder MLP parameters are trainable.
    """
    logger.info(f"Loading base model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float32,
        device_map  = "auto",
    )
    model.config.use_cache = False

    prefix_config = PrefixTuningConfig(
        task_type          = TaskType.CAUSAL_LM,
        num_virtual_tokens = NUM_VIRTUAL_TOKENS,   # 20 — from config.yaml
        # prefix_projection=True adds an MLP reparameterisation layer on top
        # of the raw prefix vectors. This stabilises training and generally
        # gives better results than raw prefix vectors (prefix_projection=False).
        prefix_projection  = True,
    )

    model = get_peft_model(model, prefix_config)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params         : {total_params:,}")
    logger.info(f"Trainable params     : {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    logger.info(f"Virtual tokens       : {NUM_VIRTUAL_TOKENS}")

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
    """
    Prefix Tuning training args.

    Key difference from LoRA:
    - gradient_checkpointing=False
        Prefix Tuning is incompatible with gradient checkpointing in the
        current PEFT implementation. The prefix vectors need to be in the
        computation graph during the backward pass, which conflicts with
        activation recomputation. Memory usage is still low because
        almost no parameters are being updated.
    """
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
        gradient_checkpointing      = False,       # incompatible with prefix tuning
        fp16                        = True,
        optim                       = "adamw_torch_fused",
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
    """Run Prefix Tuning fine-tuning."""
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, tokenizer           = load_model_and_tokenizer()
    train_dataset, val_dataset = load_dataset()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        padding            = True,
        pad_to_multiple_of = 8,
    )

    trainer = Trainer(
        model            = model,
        args             = build_training_args(),
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        processing_class = tokenizer,
        data_collator    = data_collator,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting Prefix Tuning …")
    logger.info(f"Virtual tokens  : {NUM_VIRTUAL_TOKENS}")
    logger.info(f"Prefix projection: True (MLP reparameterisation)")
    logger.info(f"Effective batch : {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        checkpoints = [
            os.path.join(OUTPUT_DIR, d)
            for d in os.listdir(OUTPUT_DIR)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming from: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save prefix encoder weights only — very small (~few MB)
    logger.info("Saving prefix adapter …")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"Training complete.")
    logger.info(f"Adapter saved to : {OUTPUT_DIR}")
    logger.info(f"Train loss       : {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    train()
