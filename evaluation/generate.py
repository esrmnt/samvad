"""
samvad/evaluation/generate.py

Runs inference on the held-out test set using all four fine-tuned models
and saves the generated counterspeech responses to CSV files.

Each model gets its own output file:
    dataset/results/{model_slug}/generated/full_finetune.csv
    dataset/results/{model_slug}/generated/lora.csv
    dataset/results/{model_slug}/generated/qlora.csv
    dataset/results/{model_slug}/generated/prefix_tuning.csv

Each CSV has columns:
    hate_speech | cs_type | reference | generated

Usage (via main.py):
    python main.py --generate

Usage (directly):
    python evaluation/generate.py
"""

import logging
import os

import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import config
from loaders.preprocess import build_prompt

logger = logging.getLogger(__name__)

MODEL_ID        = config.get("model.id")
PROCESSED_DIR   = config.processed_data_dir()
GENERATED_DIR   = config.generated_dir()

MAX_NEW_TOKENS  = int(config.get("generation.max_new_tokens"))
TEMPERATURE     = float(config.get("generation.temperature"))
TOP_P           = float(config.get("generation.top_p"))
DO_SAMPLE       = bool(config.get("generation.do_sample"))

# Model registry — name → checkpoint path and loading strategy
MODELS = {
    "full_finetune" : {"path": config.checkpoint_dir("full-finetune"),  "type": "full"},
    "lora"          : {"path": config.checkpoint_dir("lora"),           "type": "peft"},
    "qlora"         : {"path": config.checkpoint_dir("qlora"),          "type": "qlora"},
    "prefix_tuning" : {"path": config.checkpoint_dir("prefix-tuning"),  "type": "peft"},
}


def load_tokenizer(checkpoint_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation
    return tokenizer


def load_full_model(checkpoint_path: str):
    """Load a fully fine-tuned model (all weights saved)."""
    logger.info(f"Loading full model from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model


def load_peft_model(checkpoint_path: str):
    """Load base model + PEFT adapter (LoRA or Prefix Tuning)."""
    logger.info(f"Loading PEFT adapter from: {checkpoint_path}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model


def load_qlora_model(checkpoint_path: str):
    """Load base model in 4-bit + QLoRA adapter."""
    logger.info(f"Loading QLoRA adapter from: {checkpoint_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = config.get("qlora.quant_type"),
        bnb_4bit_use_double_quant = bool(config.get("qlora.double_quant")),
        bnb_4bit_compute_dtype    = torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    return model


def load_model(name: str, info: dict):
    """Dispatch to the correct loader based on model type."""
    if info["type"] == "full":
        return load_full_model(info["path"])
    elif info["type"] == "qlora":
        return load_qlora_model(info["path"])
    else:
        return load_peft_model(info["path"])


def generate_batch(
    model,
    tokenizer,
    hate_speeches: list,
    cs_types: list,
    batch_size: int = 4,
) -> list:
    """
    Generate counterspeech responses for a list of inputs.
    Processes in batches to avoid OOM on the test set.
    """
    all_generated = []
    device = next(model.parameters()).device

    system_prompt    = config.get("prompts.system_prompt")
    assistant_marker = config.get("prompts.assistant_marker")

    for i in range(0, len(hate_speeches), batch_size):
        batch_hate   = hate_speeches[i : i + batch_size]
        batch_cstype = cs_types[i : i + batch_size]

        # Build inference prompts — no answer included
        prompts = [
            build_prompt(h, c, system_prompt, assistant_marker)
            for h, c in zip(batch_hate, batch_cstype)
        ]

        inputs = tokenizer(
            prompts,
            return_tensors    = "pt",
            padding           = True,
            truncation        = True,
            max_length        = 256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                temperature    = TEMPERATURE,
                top_p          = TOP_P,
                do_sample      = DO_SAMPLE,
                pad_token_id   = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output[prompt_len:]
            decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            all_generated.append(decoded)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Generated {min(i + batch_size, len(hate_speeches))}/{len(hate_speeches)}")

    return all_generated


def generate(model_names: list = None) -> None:
    """
    Run generation for all (or selected) models on the test split.

    Args:
        model_names: List of model names to generate for.
                     Defaults to all four models.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # Load test split — we need the raw text columns, not tensors
    logger.info(f"Loading test split from: {PROCESSED_DIR}")
    dataset     = load_from_disk(PROCESSED_DIR)
    test_split  = dataset["test"]

    system_prompt     = config.get("prompts.system_prompt")
    assistant_marker  = config.get("prompts.assistant_marker")

    hate_speeches = test_split["hate_speech"]
    cs_types      = test_split["cs_type"]
    references    = test_split["counterspeech"]

    logger.info(f"Test samples: {len(test_split)}")

    targets = model_names or list(MODELS.keys())

    for name in targets:
        output_path = os.path.join(GENERATED_DIR, f"{name}.csv")

        # Skip if already generated
        if os.path.exists(output_path):
            logger.info(f"[{name}] Already generated — skipping. Delete file to regenerate.")
            continue

        info = MODELS[name]
        if not os.path.exists(info["path"]):
            logger.warning(f"[{name}] Checkpoint not found at {info['path']} — skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Generating: {name}")
        logger.info(f"{'='*60}")

        try:
            tokenizer = load_tokenizer(info["path"])
            model     = load_model(name, info)

            generated = generate_batch(
                model        = model,
                tokenizer    = tokenizer,
                hate_speeches= list(hate_speeches),
                cs_types     = list(cs_types),
                batch_size   = 4,
            )

            # Save to CSV
            df = pd.DataFrame({
                "hate_speech" : hate_speeches,
                "cs_type"     : cs_types,
                "reference"   : references,
                "generated"   : generated,
            })
            df.to_csv(output_path, index=False)
            logger.info(f"[{name}] Saved {len(df)} rows → {output_path}")

        except Exception as e:
            logger.error(f"[{name}] Generation failed: {e}")
            raise

        finally:
            # Free GPU memory before loading next model
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    generate()
