"""
samvad/data/preprocess.py

Data preprocessing pipeline: loads IntentCONANv2 from HuggingFace, formats
each sample into ChatML prompts, and saves train/val/test splits as Arrow datasets.
"""

import os
import logging
import statistics
from typing import Dict, Tuple

import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, concatenate_datasets, load_dataset

from config import config

logger = logging.getLogger(__name__)

def build_prompt(
    hate_speech: str,
    cs_type: str,
    system_prompt: str,
    assistant_marker: str,
    counterspeech: str = None,
) -> str:
    """Build a ChatML-formatted prompt for Qwen2-Instruct.

    Args:
        hate_speech: The hate speech text
        cs_type: Counter-speech type/strategy
        system_prompt: System prompt to use
        assistant_marker: Assistant marker token
        counterspeech: The counter-response (None for inference prompts)

    Returns:
        ChatML-formatted prompt string
    """
    user_content = (
        f"Hate speech: {hate_speech.strip()}\n"
        f"Response strategy: {cs_type.strip()}\n"
        f"Generate a counterspeech response:"
    )

    prefix = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"{assistant_marker}"
    )

    if counterspeech is None:
        return prefix
    return prefix + counterspeech.strip() + "<|im_end|>"


def add_prompt(example: dict, system_prompt: str, assistant_marker: str) -> dict:
    """Convert raw dataset row into prompt columns.
    
    Handles both 'hatespeech' and 'hate_speech' column name variants.
    """
    hate = example.get("hatespeech") or example.get("hate_speech", "")
    cstype = example.get("csType") or example.get("cs_type", "")
    cs = example.get("counterspeech", "")

    return {
        "prompt": build_prompt(hate, cstype, system_prompt, assistant_marker, cs),
        "prompt_no_ans": build_prompt(hate, cstype, system_prompt, assistant_marker),
        "hate_speech": hate,
        "cs_type": cstype,
        "counterspeech": cs,
    }


def tokenize(
    batch: dict,
    tokenizer: AutoTokenizer,
    assistant_marker: str,
    max_length: int,
) -> dict:
    """Tokenize batch with attention mask labels for loss masking.
    
    Loss is computed only on the assistant's reply — system prompt, user
    turn, and assistant marker are masked to -100.
    """
    encodings = tokenizer(
        batch["prompt"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    labels = []
    for input_ids, prompt_text in zip(encodings["input_ids"], batch["prompt"]):
        prefix = prompt_text.split(assistant_marker)[0] + assistant_marker
        prefix_ids = tokenizer(
            prefix, truncation=True, max_length=max_length
        )["input_ids"]
        n_masked = len(prefix_ids)

        label_ids = [-100] * n_masked + input_ids[n_masked:]
        label_ids = (label_ids + [-100] * max_length)[:max_length]
        labels.append(np.array(label_ids, dtype=np.int64))

    encodings["labels"] = labels
    return encodings


def load_full_dataset(dataset_id: str):
    """Load and normalize to a single flat split.
    
    Handles different dataset split structures from HuggingFace.
    """
    raw = load_dataset(dataset_id)
    logger.info(f"Splits found: {list(raw.keys())}")
    if list(raw.keys()):
        first_split = list(raw.keys())[0]
        logger.info(f"Columns: {raw[first_split].column_names}")

    if "train" in raw and "test" in raw:
        return concatenate_datasets([raw["train"], raw["test"]])
    return raw["train"] if "train" in raw else raw[list(raw.keys())[0]]


def make_splits(
    full_dataset,
    test_size: float,
    val_size: float,
    seed: int,
) -> DatasetDict:
    """Create train/val/test splits with proper test isolation.
    
    Test set is held out first to prevent leakage.
    """
    holdout = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_val = holdout["train"].train_test_split(
        test_size=val_size / (1 - test_size), seed=seed
    )
    return DatasetDict({
        "train": train_val["train"],
        "val": train_val["test"],
        "test": holdout["test"],
    })


def print_length_stats(
    tokenized_dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> None:
    """Log token length statistics for train split."""
    pad_id = tokenizer.pad_token_id
    lengths = [
        sum(1 for t in ids if t != pad_id)
        for ids in tokenized_dataset["train"]["input_ids"]
    ]
    if lengths:
        logger.info(
            f"Token length stats (train): "
            f"mean={statistics.mean(lengths):.0f}, "
            f"median={statistics.median(lengths):.0f}, "
            f"max={max(lengths)}, "
            f"truncated (>{max_length})={sum(l > max_length for l in lengths)}"
        )


def preprocess(output_dir: str = None, preview: bool = False) -> None:
    """Run the data preprocessing pipeline.
    
    Args:
        output_dir: Output directory for processed data (uses config default if None)
        preview: If True, show sample prompts and exit without saving
    """
    # Load configuration
    system_prompt = config.get("prompts.system_prompt")
    assistant_marker = config.get("prompts.assistant_marker")
    max_length = config.get("data.max_length")
    test_size = config.get("data.test_size")
    val_size = config.get("data.val_size")
    seed = config.get("data.seed")
    dataset_id = config.get("dataset.id")
    model_id = config.get("model.id")
    processed_data_dir = config.get("paths.processed_data_dir")

    # Load dataset
    logger.info(f"Loading dataset: {dataset_id}")
    full = load_full_dataset(dataset_id)
    logger.info(f"Total samples: {len(full)}")

    # Split data
    logger.info(
        f"Splitting — train/val/test "
        f"({int((1-test_size-val_size)*100)}/{int(val_size*100)}/{int(test_size*100)}%)"
    )
    splits = make_splits(full, test_size, val_size, seed)
    for name, ds in splits.items():
        logger.info(f"{name}: {len(ds)} samples")

    # Build prompts
    logger.info("Building prompts …")
    splits = splits.map(
        lambda example: add_prompt(example, system_prompt, assistant_marker),
        desc="Formatting prompts",
    )

    if preview:
        logger.info("Sample prompts (from train split):")
        for i in range(min(3, len(splits["train"]))):
            logger.info(f"\n[Sample {i + 1}]\n{splits['train'][i]['prompt']}\n" + "─" * 60)
        return

    # Tokenize
    logger.info(f"Tokenising with {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    cols_to_keep = {"prompt", "prompt_no_ans", "hate_speech", "cs_type", "counterspeech"}
    cols_to_drop = [c for c in splits["train"].column_names if c not in cols_to_keep]

    tokenized = splits.map(
        lambda batch: tokenize(batch, tokenizer, assistant_marker, max_length),
        batched=True,
        remove_columns=cols_to_drop,
        desc="Tokenising",
    )
    tokenized.set_format("torch")

    # Log statistics
    print_length_stats(tokenized, tokenizer, max_length)

    # Save
    final_output_dir = output_dir or processed_data_dir
    logger.info(f"Saving to {final_output_dir} …")
    os.makedirs(final_output_dir, exist_ok=True)
    tokenized.save_to_disk(final_output_dir)

    # Save sample for inspection
    splits["train"].select(range(min(500, len(splits["train"])))).to_csv(
        os.path.join(final_output_dir, "train_sample.csv"), index=False
    )

    logger.info(f"Done. Directory contents: {sorted(os.listdir(final_output_dir))}")

    print_length_stats(tokenized, tokenizer, max_length)