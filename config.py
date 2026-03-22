"""
samvad/config.py

Configuration for all project-wide settings. Every other file imports from here.
To override for a specific run, pass args via the CLI in each script.
"""

# Model & dataset identifiers from HuggingFace Hub

MODEL_ID   = "Qwen/Qwen2-0.5B-Instruct"
DATASET_ID = "Aswini123/IntentCONANv2"

# Data 

MAX_LENGTH = 256                        # max tokens per sample — covers 100% of IntentCONANv2
VAL_SIZE   = 0.1                        # fraction of train split used for validation
TEST_SIZE  = 0.1                        # fraction of full dataset held out for final evaluation
SEED       = 42

# Paths

PROCESSED_DATA_DIR = "./data/processed"
CHECKPOINTS_DIR    = "./checkpoints"
RESULTS_DIR        = "./results"

# Prompt components (shared across training and inference prompts for consistency)

SYSTEM_PROMPT = (
    "You are a counterspeech assistant. Given a hate speech message and a "
    "desired response strategy, generate a constructive and non-toxic counter "
    "response."
)

ASSISTANT_MARKER = "<|im_start|>assistant\n"

# Training (shared defaults across all fine-tuning methods)

LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
BATCH_SIZE       = 1                    # keep at 1 for full FT on 12GB VRAM
GRAD_ACCUM_STEPS = 8                    # effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
WARMUP_RATIO     = 0.05
LR_SCHEDULER     = "cosine"
LOGGING_STEPS    = 10
SAVE_STEPS       = 100

# LoRA 

LORA_R          = 16                    # rank — higher = more capacity, more memory
LORA_ALPHA      = 32                    # scaling factor, typically 2× rank
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# QLoRA (inherits LoRA settings above, adds quantisation) 

QLORA_BITS           = 4                # quantise base model to 4-bit
QLORA_DOUBLE_QUANT   = True             # quantise the quantisation constants too
QLORA_QUANT_TYPE     = "nf4"            # normal float 4 — best for LLM weights

# Prefix tuning

PREFIX_NUM_VIRTUAL_TOKENS = 20          # number of learnable prefix tokens

# Generation (used at inference / evaluation time) 

MAX_NEW_TOKENS  = 128
TEMPERATURE     = 0.7
TOP_P           = 0.9
DO_SAMPLE       = True