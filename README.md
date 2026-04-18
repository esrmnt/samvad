# samvad

Counterspeech generation with fine-tuned language models. This project provides a complete pipeline for preprocessing hate speech datasets and training multiple fine-tuning strategies (standard fine-tuning, LoRA, and QLoRA) for generating effective counter-responses.

## Quick Start

### Setup

One-time environment setup:

```bash
cd samvad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Project

Activate the environment and run:

```bash
source venv/bin/activate
python main.py
```

## Workflows

### Data Preprocessing

Preprocess the dataset from the main entry point:

```bash
# Preview sample prompts without saving
python main.py --preprocess --preview

# Run full preprocessing pipeline
python main.py --preprocess

# Save to custom output directory
python main.py --preprocess --output_dir ./custom_data_path
```

Or run preprocessing directly:

```bash
python data/preprocess.py --preview
python data/preprocess.py
python data/preprocess.py --output_dir ./custom_data_path
```

Or import in your code:

```python
from loaders.preprocess import preprocess

preprocess()                                # uses config defaults
preprocess(output_dir="./my_data")          # custom output
preprocess(preview=True)                    # preview mode
```

**Preprocessing pipeline:**
1. Load dataset from HuggingFace Hub (IntentCONANv2)
2. Split into train/validation/test sets
3. Build ChatML-formatted prompts
4. Tokenize with label masking
5. Save as Arrow datasets to `dataset/processed/`

### Model Training

Train the model using different fine-tuning strategies:

```bash
# Full fine-tuning (all parameters updated)
python main.py --train_full

# LoRA fine-tuning (16-rank adaptation layers)
python main.py --train_lora

# QLoRA fine-tuning (quantized LoRA with 4-bit)
python main.py --train_qlora
```


**Training outputs:**
- Checkpoints saved to `dataset/checkpoints/{method}/`
- Training results and metrics in `dataset/results/`
- Trained adapters (LoRA) loadable with Hugging Face `peft`

## Configuration

All project settings are managed in a single `config/config.yaml` file, globally accessible using a singleton pattern.

### Accessing Configuration

```python
from config import config

# Get values with dot notation
batch_size = config.get("training.batch_size")
model_id = config.get("model.id")
learning_rate = config.get("training.learning_rate", default=1e-4)

# Access entire sections as dictionaries
training_config = config.training
data_config = config.data
lora_config = config.lora
```

### Configuration Structure

The `config/config.yaml` file is organized into sections:

- **model**: Model identifier (default: Qwen/Qwen2-0.5B-Instruct)
- **dataset**: Dataset identifier (default: Aswini123/IntentCONANv2)
- **data**: Data processing parameters (max_length, train/val/test splits, random seed)
- **paths**: Directory paths for artifacts, checkpoints, and results
- **prompts**: System prompts and markers for conversational formatting
- **training**: Hyperparameters (learning rate, epochs, batch size, warmup, scheduler)
- **lora**: LoRA configuration (rank, alpha, target modules)
- **qlora**: QLoRA quantization settings

### Adding New Configuration

Add new values to `config/config.yaml` and access via dot notation anywhere in your code.

Example:

```yaml
# config/config.yaml
inference:
  max_new_tokens: 256
  temperature: 0.7
```

```python
# Your code
max_tokens = config.get("inference.max_new_tokens")
temp = config.get("inference.temperature", default=0.7)
```

## Project Structure

```
samvad/
├── main.py                 # CLI entry point - orchestrates all workflows
├── config/
│   ├── __init__.py        # Exports global config instance
│   ├── config.py          # Configuration loader with dot-notation access
│   └── config.yaml        # All project settings (EDIT THIS)
├── loaders/
│   ├── __init__.py
│   └── preprocess.py      # Data preprocessing and dataset loading
├── training/
│   ├── full_finetune.py   # Full parameter fine-tuning
│   ├── lora.py            # LoRA fine-tuning
│   └── qlora.py           # QLoRA quantized fine-tuning
├── dataset/               # Generated dataset and checkpoints
│   ├── processed/         # Preprocessed Arrow datasets
│   ├── checkpoints/       # Saved model checkpoints
│   └── results/           # Training metrics and results
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Installation from Source

```bash
pip install -e .
```

This installs the project in editable mode with all dependencies.

## Architecture

The project follows a clean, modular design:

- **`main.py`**: CLI orchestrator that parses arguments and delegates to workflow functions
- **`config.py`**: Singleton configuration loader with dot-notation access from any module
- **`loaders/preprocess.py`**: Data loading, preprocessing, and Arrow dataset generation
- **`training/`**: Modular training implementations (full, LoRA, QLoRA) using HuggingFace Trainer
- **`dataset/`**: Artifact storage (processed data, model checkpoints, training results)

Each workflow module can be:
- Called from `main.py` (CLI mode)
- Run directly (standalone mode)
- Imported and called from other code (library mode)

This design enables flexibility whether you're using samvad as a CLI tool or integrating it into a larger codebase.

## Fine-Tuning Methods

### Full Fine-Tuning

Updates all model parameters. Baseline approach for comparison.

- Memory intensive but often produces best results
- Use when you have sufficient GPU memory
- Training: `python main.py --train_full`

### LoRA (Low-Rank Adaptation)

Adds learnable low-rank matrices to attention layers. Recommended for most use cases.

- 10-100x fewer trainable parameters
- Fast training and inference with minimal latency
- Easily combine multiple LoRA adapters
- Training: `python main.py --train_lora`
- Default config: rank=16, alpha=32

### QLoRA (Quantized LoRA)

Combines 4-bit quantization with LoRA for extreme parameter efficiency.

- Fits large models on consumer GPUs
- Minimal performance degradation
- Training: `python main.py --train_qlora`
- Ideal for resource-constrained environments

## Dataset and Model

By default, samvad uses:

- **Model**: Qwen/Qwen2-0.5B-Instruct (500M parameters)
- **Dataset**: Aswini123/IntentCONANv2 (hate speech + counter-speech pairs)

To use different models or datasets, update `config/config.yaml`:

```yaml
model:
  id: "meta-llama/Llama-2-7b"  # Any model from Hugging Face Hub
dataset:
  id: "your-username/your-dataset"
```

## License

Licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1). See [LICENSE](LICENSE) for details.