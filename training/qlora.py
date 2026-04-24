"""
samvad/training/qlora.py

QLoRA fine-tuning for decoder-only (causal) language models on IntentCONANv2.

Tested with Qwen, but designed to support other architectures (e.g., LLaMA, Mistral) with appropriate configuration (e.g., target modules).

QLoRA = LoRA + 4-bit quantization of the base model.
Three innovations from the QLoRA paper (Dettmers et al. 2023):
  1. NF4 (NormalFloat4) — optimal quantization for normally distributed weights
  2. Double quantization — quantize the quantization constants (~0.37 bits saved/param)
  3. Paged optimizer — offloads optimizer states to CPU RAM during memory spikes

Usage (via main.py):
    python main.py --train --method qlora --model qwen

Usage (directly):
    python training/qlora.py
"""

import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    EarlyStoppingCallback,
)

from config import config
from .common import (
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

LORA_R = int(config.get("lora.r"))
LORA_ALPHA = int(config.get("lora.alpha"))
LORA_DROPOUT = float(config.get("lora.dropout"))
LORA_TARGETS = config.get("lora.target_modules")

QLORA_BITS = int(config.get("qlora.bits"))
DOUBLE_QUANT = bool(config.get("qlora.double_quant"))
QUANT_TYPE = config.get("qlora.quant_type")

RUN_NAME = "qlora"
OUTPUT_DIR = config.checkpoint_dir(RUN_NAME)



def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_use_double_quant=DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model_and_tokenizer():
    """Load base model in 4-bit NF4 with LoRA adapters."""
    tokenizer = load_tokenizer(MODEL_ID)
    
    logger.info(f"Loading model in {QLORA_BITS}-bit: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(),
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for k-bit training before adding LoRA
    model = prepare_model_for_kbit_training(model)

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
    logger.info(f"Quant type      : {QUANT_TYPE}")
    logger.info(f"Double quant    : {DOUBLE_QUANT}")

    return model, tokenizer


def train(output_dir: str = None) -> None:
    """Run QLoRA fine-tuning.
    
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
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
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
        quant_bits=f"{QLORA_BITS}-bit {QUANT_TYPE}",
        double_quant=DOUBLE_QUANT,
        rank_r=LORA_R,
    )

    last_checkpoint = find_last_checkpoint(final_output_dir)
    if last_checkpoint:
        logger.info(f"Resuming from: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving QLoRA adapter …")
    model.save_pretrained(final_output_dir)
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
