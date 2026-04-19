"""
samvad/training/prefix_tuning.py

Prefix Tuning for causal language models (e.g., Qwen, LLaMA).
Tested with Qwen, but designed to support other architectures (e.g., LLaMA, Mistral) with appropriate configuration (e.g., target modules).

Learns a small set of virtual tokens while keeping the base model frozen.
"""

import logging
import os
import json
import shutil

import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
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

ROOT_ARTIFACTS = {
    "README.md",
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "trainer_state.json",
}


def load_model_and_tokenizer():
    """Load base model and attach Prefix Tuning adapters."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model(MODEL_ID, torch_dtype=torch.float16)

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prefix_projection=True,
    )

    model = get_peft_model(model, prefix_config)
    log_model_params(model)

    return model, tokenizer


def has_final_artifacts(output_dir: str) -> bool:
    """Return True when the method root has a loadable prefix adapter."""
    return (
        os.path.exists(os.path.join(output_dir, "adapter_config.json"))
        and (
            os.path.exists(os.path.join(output_dir, "adapter_model.safetensors"))
            or os.path.exists(os.path.join(output_dir, "adapter_model.bin"))
        )
        and os.path.exists(os.path.join(output_dir, "tokenizer_config.json"))
    )


def checkpoint_is_complete(checkpoint_path: str) -> bool:
    """Return True when a checkpoint contains adapter and tokenizer artifacts."""
    if not checkpoint_path:
        return False
    return has_final_artifacts(checkpoint_path)


def checkpoint_training_finished(checkpoint_path: str) -> bool:
    """Return True when trainer state says the checkpoint reached max steps."""
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_path):
        return False

    with open(state_path) as f:
        state = json.load(f)

    max_steps = state.get("max_steps")
    global_step = state.get("global_step", 0)
    return bool(max_steps and global_step >= max_steps)


def export_checkpoint_artifacts(checkpoint_path: str, output_dir: str) -> None:
    """Copy loadable adapter/tokenizer artifacts from a checkpoint to the method root."""
    if not checkpoint_is_complete(checkpoint_path):
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Exporting Prefix artifacts from {checkpoint_path} to {output_dir}")

    for filename in ROOT_ARTIFACTS:
        src = os.path.join(checkpoint_path, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def train(output_dir: str = None) -> None:
    """Run Prefix Tuning fine-tuning."""
    final_output_dir = output_dir or OUTPUT_DIR
    os.makedirs(final_output_dir, exist_ok=True)

    last_checkpoint = find_last_checkpoint(final_output_dir)
    if (
        last_checkpoint
        and not has_final_artifacts(final_output_dir)
        and checkpoint_training_finished(last_checkpoint)
    ):
        export_checkpoint_artifacts(last_checkpoint, final_output_dir)
        if has_final_artifacts(final_output_dir):
            logger.info(f"Prefix adapter already available at: {final_output_dir}")
            return

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
        load_best_model_at_end=True,
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
        virtual_tokens=NUM_VIRTUAL_TOKENS,
    )

    if last_checkpoint:
        logger.info(f"Resuming from: {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info("Saving Prefix adapter …")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log_training_complete(final_output_dir, metrics.get("train_loss"))
    logger.info(f"Train loss : {metrics.get('train_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    train()
