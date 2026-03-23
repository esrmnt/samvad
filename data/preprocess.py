"""
samvad/data/preprocess.py

Loads IntentCONANv2 from HuggingFace, formats each sample into a
structured prompt, and saves train / val / test splits as Arrow datasets.

Usage:
    python data/preprocess.py                        # saves to default path
    python data/preprocess.py --output_dir /my/path  # custom output dir
    python data/preprocess.py --preview              # print 3 samples and exit
"""

import os
import argparse
import statistics

from transformers import AutoTokenizer
from datasets import DatasetDict, concatenate_datasets, load_dataset

from config import (
    ASSISTANT_MARKER,
    DATASET_ID,
    MAX_LENGTH,
    MODEL_ID,
    PROCESSED_DATA_DIR,
    SEED,
    SYSTEM_PROMPT,
    TEST_SIZE,
    VAL_SIZE,
)


def build_prompt(hate_speech: str, cs_type: str, counterspeech: str = None) -> str:
    """
    Build a ChatML-formatted prompt for Qwen2-Instruct.

    counterspeech=None  →  inference prompt  (no answer, model generates it)
    counterspeech=str   →  training prompt   (answer included for supervision)

    Both share the exact same prefix so training and inference are consistent.
    """
    user_content = (
        f"Hate speech: {hate_speech.strip()}\n"
        f"Response strategy: {cs_type.strip()}\n"
        f"Generate a counterspeech response:"
    )

    prefix = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"{ASSISTANT_MARKER}"
    )

    if counterspeech is None:
        return prefix                                           # used for inference
    return prefix + counterspeech.strip() + "<|im_end|>"        # used for training


def add_prompt(example: dict) -> dict:
    """
    Map function: converts one raw dataset row into prompt columns.
    Handles both 'hatespeech' and 'hate_speech' column name variants.
    """
    hate   = example.get("hatespeech") or example.get("hate_speech", "")
    cstype = example.get("csType")     or example.get("cs_type", "")
    cs     = example.get("counterspeech", "")

    return {
        "prompt":        build_prompt(hate, cstype, cs),        # training prompt
        "prompt_no_ans": build_prompt(hate, cstype),            # inference prompt
        "hate_speech":   hate,
        "cs_type":       cstype,
        "counterspeech": cs,
    }


def tokenize(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenise a batch of training prompts with -100 label masking.

    Loss is computed only on the assistant's reply — system prompt, user
    turn, and assistant marker are all masked to -100 so PyTorch's
    cross-entropy skips them entirely.
    """
    encodings = tokenizer(
        batch["prompt"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = []
    for input_ids, prompt_text in zip(encodings["input_ids"], batch["prompt"]):
        prefix     = prompt_text.split(ASSISTANT_MARKER)[0] + ASSISTANT_MARKER
        prefix_ids = tokenizer(prefix, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        n_masked   = len(prefix_ids)

        label_ids = [-100] * n_masked + input_ids[n_masked:]
        label_ids = (label_ids + [-100] * MAX_LENGTH)[:MAX_LENGTH]
        labels.append(label_ids)

    encodings["labels"] = labels
    return encodings


def load_full_dataset(dataset_id: str):
    """
    Load and normalise to a single flat split regardless of how the
    dataset was originally packaged on HuggingFace.
    """
    raw = load_dataset(dataset_id)
    print(f"      Splits found : {list(raw.keys())}")
    print(f"      Columns      : {raw[list(raw.keys())[0]].column_names}")

    if "train" in raw and "test" in raw:
        return concatenate_datasets([raw["train"], raw["test"]])
    return raw["train"] if "train" in raw else raw[list(raw.keys())[0]]


def make_splits(full_dataset) -> DatasetDict:
    """
    Carve out test first, then split remainder into train / val.
    Test is isolated before any other decision — no leakage.
    """
    holdout   = full_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    train_val = holdout["train"].train_test_split(
        test_size=VAL_SIZE / (1 - TEST_SIZE), seed=SEED
    )
    return DatasetDict({
        "train": train_val["train"],
        "val":   train_val["test"],
        "test":  holdout["test"],
    })


def print_length_stats(tokenized_dataset, tokenizer):
    """Print non-padding token length stats for the train split."""
    pad_id  = tokenizer.pad_token_id
    lengths = [
        sum(1 for t in ids if t != pad_id)
        for ids in tokenized_dataset["train"]["input_ids"]
    ]
    print("\nToken length stats (train split):")
    print(f"  mean      : {statistics.mean(lengths):.0f}")
    print(f"  median    : {statistics.median(lengths):.0f}")
    print(f"  max       : {max(lengths)}")
    print(f"  truncated (>{MAX_LENGTH}): {sum(l > MAX_LENGTH for l in lengths)}")


def main(args):
    # 1. Load
    print(f"Loading dataset: {DATASET_ID}")
    full = load_full_dataset(DATASET_ID)
    print(f"      Total samples: {len(full)}")

    # 2. Split
    print(f"Splitting — train/val/test ({int((1-TEST_SIZE-VAL_SIZE)*100)}/{int(VAL_SIZE*100)}/{int(TEST_SIZE*100)}%)")
    splits = make_splits(full)
    for name, ds in splits.items():
        print(f"      {name}: {len(ds)} samples")

    # 3. Build prompts
    print("Building prompts …")
    splits = splits.map(add_prompt, desc="Formatting prompts")

    if args.preview:
        print("\n Sample prompts (from train split):")
        for i in range(min(3, len(splits["train"]))):
            print(f"\n[Sample {i + 1}]")
            print(splits["train"][i]["prompt"])
            print("─" * 60)
        return

    # 4. Tokenise
    print(f"Tokenising with {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    cols_to_keep = {"prompt", "prompt_no_ans", "hate_speech", "cs_type", "counterspeech"}
    cols_to_drop = [c for c in splits["train"].column_names if c not in cols_to_keep]

    tokenized = splits.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=cols_to_drop,
        desc="Tokenising",
    )
    tokenized.set_format("torch")

    # 5. Save
    output_dir = args.output_dir or PROCESSED_DATA_DIR
    print(f"Saving to {output_dir} …")
    os.makedirs(output_dir, exist_ok=True)
    tokenized.save_to_disk(output_dir)

    splits["train"].select(range(min(500, len(splits["train"])))).to_csv(
        os.path.join(output_dir, "train_sample.csv"), index=False
    )

    print("\nDone. Directory contents:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")

    print_length_stats(tokenized, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to save the processed dataset (default: config.PROCESSED_DATA_DIR)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print 3 sample prompts and exit without saving"
    )
    main(parser.parse_args())