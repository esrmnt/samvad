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

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID          = config.get("model.id")
PROCESSED_DIR     = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR   = config.get("paths.checkpoints_dir")
LEARNING_RATE     = config.get("training.learning_rate")
NUM_EPOCHS        = config.get("training.num_epochs")
BATCH_SIZE        = config.get("training.batch_size")
GRAD_ACCUM_STEPS  = config.get("training.grad_accum_steps")
WARMUP_RATIO      = config.get("training.warmup_ratio")
LR_SCHEDULER      = config.get("training.lr_scheduler")
LOGGING_STEPS     = config.get("training.logging_steps")
SAVE_STEPS        = config.get("training.save_steps")
MAX_LENGTH        = config.get("data.max_length")

RUN_NAME          = "full-finetune"
OUTPUT_DIR        = os.path.join(CHECKPOINTS_DIR, RUN_NAME)


# ── Model & tokenizer ──────────────────────────────────────────────────────────

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
        torch_dtype=torch.float16,     # fp16 — halves VRAM vs float32
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


# ── Dataset ────────────────────────────────────────────────────────────────────

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


# ── Training arguments ─────────────────────────────────────────────────────────

def build_training_args() -> TrainingArguments:
    """
    Build TrainingArguments configured for full FT on a 12GB GPU.

    Key memory settings:
    - fp16=True                     : half precision forward + backward
    - gradient_checkpointing=True   : recompute activations, don't store them
    - per_device_train_batch_size=1
      + gradient_accumulation_steps=8 : effective batch = 8, but only 1
                                        sample in VRAM at a time
    """
    return TrainingArguments(
        # ── Output & logging ──────────────────────────────────────────────────
        output_dir          = OUTPUT_DIR,
        run_name            = RUN_NAME,
        logging_dir         = os.path.join(OUTPUT_DIR, "logs"),
        logging_strategy    = "steps",
        logging_steps       = LOGGING_STEPS,
        report_to           = "none",          # local only — no W&B

        # ── Training schedule ─────────────────────────────────────────────────
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = LR_SCHEDULER,
        warmup_ratio                = WARMUP_RATIO,

        # ── Batch & memory ────────────────────────────────────────────────────
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM_STEPS,
        gradient_checkpointing      = True,
        fp16                        = True,
        optim                       = "adamw_torch_fused",

        # ── Evaluation & checkpointing ────────────────────────────────────────
        eval_strategy               = "steps",
        eval_steps                  = SAVE_STEPS,
        save_strategy               = "steps",
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 2,        # keep only 2 checkpoints on disk
        load_best_model_at_end      = True,     # loads best val loss checkpoint
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,    # lower loss = better

        # ── Reproducibility ───────────────────────────────────────────────────
        seed                        = 42,
        data_seed                   = 42,

        # ── Misc ──────────────────────────────────────────────────────────────
        remove_unused_columns       = False,    # we manage columns ourselves
        group_by_length             = True,     # batches similar lengths → less padding waste
    )


# ── Trainer ────────────────────────────────────────────────────────────────────

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

    # Data collator — handles dynamic padding within each batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        padding            = True,
        pad_to_multiple_of = 8,    # aligns to tensor cores for fp16 speedup
    )

    training_args = build_training_args()

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = dataset["train"],
        eval_dataset  = dataset["val"],
        tokenizer     = tokenizer,
        data_collator = data_collator,
        callbacks     = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
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

    # ── Save final model ───────────────────────────────────────────────────────
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