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
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from config import config

logger = logging.getLogger(__name__)


MODEL_ID          = config.get("model.id")
PROCESSED_DIR     = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR   = config.get("paths.checkpoints_dir")
LEARNING_RATE     = float(config.get("training.learning_rate"))
NUM_EPOCHS        = int(config.get("training.num_epochs"))
BATCH_SIZE        = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS  = int(config.get("training.grad_accum_steps"))
WARMUP_RATIO      = float(config.get("training.warmup_ratio"))
LOGGING_STEPS     = int(config.get("training.logging_steps"))
SAVE_STEPS        = int(config.get("training.save_steps"))
MAX_LENGTH        = int(config.get("data.max_length"))
LR_SCHEDULER      = config.get("training.lr_scheduler")

RUN_NAME          = "full-finetune"
OUTPUT_DIR        = os.path.join(CHECKPOINTS_DIR, RUN_NAME)


def load_model_and_tokenizer():
    """
    Load base model and tokenizer.

    fp16=True halves memory usage — essential for fitting full FT
    on a 12GB GPU. gradient_checkpointing is enabled in TrainingArguments
    which recomputes activations during backward pass instead of storing
    them, cutting VRAM by ~40% at the cost of ~20% slower training.
    """
    logger.info(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,     # ← float32 on load, trainer handles fp16
        device_map="auto",             # places model on GPU automatically
    )

    # Disable cache during training — incompatible with gradient checkpointing
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    return model, tokenizer


def load_dataset():
    """Load preprocessed Arrow dataset from disk."""
    logger.info(f"Loading dataset from: {PROCESSED_DIR}")

    if not os.path.exists(PROCESSED_DIR):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DIR}. "
            "Run: python main.py --preprocess"
        )

    dataset = load_from_disk(PROCESSED_DIR)
    logger.info(f"Train : {len(dataset['train']):,} samples")
    logger.info(f"Val   : {len(dataset['val']):,} samples")
    logger.info(f"Test  : {len(dataset['test']):,} samples")
    return dataset


def build_training_args() -> TrainingArguments:
    """
    Build TrainingArguments configured for full FT on a 12GB GPU.

    Key memory settings:
    - fp16=True                     : half precision forward + backward
    - gradient_checkpointing=True   : recompute activations, don't store them
    - per_device_train_batch_size=1
      + gradient_accumulation_steps=8 : effective batch = 8, but only 1
                                        sample in VRAM at a time

    Changes from previous version:
    - warmup_ratio  → warmup_steps  (warmup_ratio deprecated in v5.2)
    - logging_dir   removed         (deprecated in v5.2; set env var
                                     TENSORBOARD_LOGGING_DIR if needed)
    """
    # Compute warmup_steps from warmup_ratio as a best-effort approximation.
    # Trainer normally derives this from the dataset size, but we can set it
    # explicitly here to avoid the deprecation warning.
    # Formula: warmup_steps = ceil(warmup_ratio * total_steps)
    #          total_steps  = ceil(train_samples / effective_batch) * epochs
    # Since we don't have dataset size here, we use a conservative fixed value.
    # Override this constant if your dataset size changes significantly.
    WARMUP_STEPS = 100  # ~10% of steps for a ~800-sample train split at eff. batch=8

    return TrainingArguments(
        # ── Output & logging ──────────────────────────────────────────────────
        output_dir          = OUTPUT_DIR,
        run_name            = RUN_NAME,
        # logging_dir is deprecated in v5.2 — set TENSORBOARD_LOGGING_DIR env
        # var instead if you need TensorBoard output:
        #   export TENSORBOARD_LOGGING_DIR=<path>
        logging_strategy    = "steps",
        logging_steps       = LOGGING_STEPS,
        report_to           = "none",          # local only — no W&B

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

        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,        # keep only 2 checkpoints on disk
        load_best_model_at_end      = True,     # loads best val loss checkpoint
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,    # lower loss = better

        seed                        = 42,
        data_seed                   = 42,

        remove_unused_columns       = False    # we manage columns ourselves
    )

def train(output_dir: str = None) -> None:
    """
    Run full fine-tuning.

    Args:
        output_dir: Override checkpoint directory (uses config default if None)
    """
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model, tokenizer, data
    model, tokenizer = load_model_and_tokenizer()
    dataset          = load_dataset()

    # Keep only tensor columns — collator can't handle raw text columns
    tensor_cols  = {"input_ids", "attention_mask", "labels"}
    train_dataset = dataset["train"].remove_columns([c for c in dataset["train"].column_names if c not in tensor_cols])
    val_dataset = dataset["val"].remove_columns([c for c in dataset["val"].column_names if c not in tensor_cols])
    train_dataset.set_format("torch")   
    val_dataset.set_format("torch")    

    # Data collator — handles dynamic padding within each batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        padding            = True,
        pad_to_multiple_of = 8,    # aligns to tensor cores for fp16 speedup
    )

    training_args = build_training_args()

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        # 'tokenizer' kwarg removed in transformers v4.47+ → use processing_class
        # See: https://huggingface.co/docs/transformers/main/en/main_classes/trainer
        processing_class = tokenizer,
        data_collator    = data_collator,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting full fine-tuning …")
    logger.info(f"Effective batch size : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"Checkpoint dir       : {OUTPUT_DIR}")

    # Resume from checkpoint if one exists
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