"""
samvad/training/qlora.py

QLoRA fine-tuning of Qwen2-0.5B-Instruct on IntentCONANv2.

QLoRA = LoRA + 4-bit quantization of the base model.
Three innovations from the QLoRA paper (Dettmers et al. 2023):
  1. NF4 (NormalFloat4) — optimal quantization for normally distributed weights
  2. Double quantization — quantize the quantization constants to save ~0.37 bits/param
  3. Paged optimizer — offloads optimizer states to CPU RAM during memory spikes

The base model is loaded in 4-bit NF4, frozen, then dequantized on-the-fly
to bfloat16 for computation. LoRA adapters train in bfloat16 alongside it.

Memory footprint: ~3-4GB vs ~8GB for standard LoRA.

Usage (via main.py):
    python main.py --train_qlora

Usage (directly):
    python training/qlora.py
"""
import os
import logging

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import config

logger = logging.getLogger(__name__)

MODEL_ID         = config.get("model.id")
PROCESSED_DIR    = config.get("paths.processed_data_dir")
CHECKPOINTS_DIR  = config.get("paths.checkpoints_dir")
LEARNING_RATE    = float(config.get("training.learning_rate"))
NUM_EPOCHS       = int(config.get("training.num_epochs"))
BATCH_SIZE       = int(config.get("training.batch_size"))
GRAD_ACCUM_STEPS = int(config.get("training.grad_accum_steps"))
WARMUP_STEPS     = 100
LOGGING_STEPS    = int(config.get("training.logging_steps"))
SAVE_STEPS       = int(config.get("training.save_steps"))
LR_SCHEDULER     = config.get("training.lr_scheduler")

LORA_R           = int(config.get("lora.r"))
LORA_ALPHA       = int(config.get("lora.alpha"))
LORA_DROPOUT     = float(config.get("lora.dropout"))
LORA_TARGETS     = config.get("lora.target_modules")

QLORA_BITS       = int(config.get("qlora.bits"))
DOUBLE_QUANT     = bool(config.get("qlora.double_quant"))
QUANT_TYPE       = config.get("qlora.quant_type")

RUN_NAME         = "qlora"
OUTPUT_DIR       = os.path.join(CHECKPOINTS_DIR, RUN_NAME)


def build_bnb_config() -> BitsAndBytesConfig:
    """
    Build the 4-bit quantisation config (BitsAndBytes).

    bnb_4bit_quant_type="nf4"
        NormalFloat4 — information-theoretically optimal for weights that
        follow a normal distribution (which pretrained LLM weights do).
        Superior to fp4 by ~1 percentage point on downstream tasks.

    bnb_4bit_use_double_quant=True
        Quantises the quantisation constants themselves, saving ~0.37
        bits/param (~3GB on a 65B model).

    bnb_4bit_compute_dtype=torch.bfloat16
        Weights are stored in 4-bit but dequantised to bfloat16 on-the-fly
        for the actual matrix multiplications. Never train in 4-bit directly.
    """
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = QUANT_TYPE,       # "nf4"
        bnb_4bit_use_double_quant = DOUBLE_QUANT,     # True
        bnb_4bit_compute_dtype    = torch.bfloat16,   # dequant target dtype
    )

def load_model_and_tokenizer():
    """
    Load base model in 4-bit NF4, prepare for k-bit training,
    then inject LoRA adapters.

    prepare_model_for_kbit_training() does three things:
      - Casts layernorm layers to float32 for stability
      - Freezes all base model parameters
      - Enables gradient checkpointing
    """
    logger.info(f"Loading model in {QLORA_BITS}-bit: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = build_bnb_config(),  # 4-bit NF4
        device_map          = "auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for k-bit training — must be called before get_peft_model()
    model = prepare_model_for_kbit_training(model)

    # Inject LoRA adapters — identical config to lora.py
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGETS,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params    : {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Quant type      : {QUANT_TYPE}")
    logger.info(f"Double quant    : {DOUBLE_QUANT}")

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
    QLoRA-specific training args.

    Key differences from full FT and LoRA:
    - fp16=False, bf16=True  : QLoRA requires bfloat16, not float16
    - optim="paged_adamw_32bit" : paged optimizer offloads states to CPU
                                   during memory spikes — essential for QLoRA
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
        gradient_checkpointing      = True,
        fp16                        = False,           # must be False for QLoRA
        bf16                        = True,            # QLoRA computes in bfloat16
        optim                       = "paged_adamw_32bit",  # paged optimizer
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
    """Run QLoRA fine-tuning."""
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

    logger.info("Starting QLoRA fine-tuning …")
    logger.info(f"Quant bits      : {QLORA_BITS}-bit {QUANT_TYPE}")
    logger.info(f"Double quant    : {DOUBLE_QUANT}")
    logger.info(f"Rank r          : {LORA_R}")
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

    # Save adapter weights only
    logger.info("Saving QLoRA adapter …")
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
