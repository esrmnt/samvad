# samvad

Counterspeech generation with fine-tuned LLMs.

## Quick Start

### Setup

One-time environment setup:

```bash
cd /Users/equo/Src/samvad
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

Or if installed as a package:

```bash
samvad
```

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

Or run preprocessing directly as a standalone module:

```bash
python data/preprocess.py --preview
python data/preprocess.py
python data/preprocess.py --output_dir ./custom_data_path
```

Or import and use the preprocessing function in your own code:

```python
from data.preprocess import preprocess

preprocess()                                # uses config defaults
preprocess(output_dir="./my_data")         # custom output
preprocess(preview=True)                    # preview mode
```

**Preprocessing steps:**
1. Load dataset from HuggingFace Hub
2. Split into train/validation/test
3. Build ChatML-formatted prompts
4. Tokenize with label masking
5. Save as Arrow datasets to `data/processed/`

## Configuration System

Configuration is managed in a single `config/config.yaml` file, globally accessible throughout the project using an industry-standard singleton pattern.

### Accessing Config from Any File

```python
from config import config

# Get nested values with dot notation
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

- **model**: Model and dataset identifiers
- **data**: Data processing parameters (max_length, splits, seed)
- **paths**: Directory paths for processed data, checkpoints, results
- **prompts**: System prompts and markers
- **training**: Training hyperparameters
- **lora**: LoRA fine-tuning configuration
- **qlora**: QLoRA quantization settings
- **prefix**: Prefix tuning parameters
- **generation**: Inference/generation parameters

### Adding New Configuration

1. Add new values to `config/config.yaml`
2. Access via dot notation: `config.get("section.key")`
3. Use `config.get(..., default=value)` for optional values

Example:

```yaml
# config/config.yaml
gpu:
  device: "cuda:0"
  fp16: true
```

```python
# Your code
device = config.get("gpu.device")
use_fp16 = config.get("gpu.fp16")
```

## Project Structure

```
samvad/
├── main.py                 # Entry point - run this to start
├── config/
│   ├── __init__.py        # Exports global config instance
│   ├── config.py          # Configuration loader
│   └── config.yaml        # All project configuration (EDIT THIS)
├── data/
│   ├── __init__.py
│   └── preprocess.py      # Data preprocessing module
├── pyproject.toml         # Project metadata and dependencies
├── LICENSE
├── .gitignore
└── README.md              # This file
```

## Troubleshooting

**Q: `ModuleNotFoundError: No module named 'yaml'`**
- A: Install via: `pip install -e .` (installs project + dependencies)

**Q: `FileNotFoundError: Config file not found`**
- A: Ensure you're in the project root directory when running

**Q: How do I run tests?**
- A: Tests can be added to a `tests/` directory and run with `pytest`

## Best Practices

✓ **DO:**
- Access config from the global instance
- Use dot notation for nested values
- Keep all config in `config/config.yaml`
- Use `config.get(..., default=value)` for optional values
- Document config keys you're using in comments
- Run preprocessing before model training

✗ **DON'T:**
- Hardcode values in your code
- Create separate config files for modules
- Store secrets in `config.yaml` (use environment variables instead)

## Architecture

**Orchestrator Pattern (Industry Standard):**

- **`main.py`**: Single CLI entry point that orchestrates all workflows
  - Parses command-line arguments
  - Loads configuration
  - Delegates to specific module functions
  
- **`data/preprocess.py`**: Specific workflow function
  - Exports `preprocess()` function for programmatic use
  - Can be run standalone (`python data/preprocess.py`)
  - Can be imported and called directly (`from data.preprocess import preprocess`)
  
- **`config/`**: Centralized configuration
  - Singleton pattern
  - Globally accessible from any module
  - YAML-based configuration file

**Benefits:**
- Clear separation of concerns
- Single responsibility per module
- Reusable functions (can be called from main, CLI, or other code)
- Easy to add new workflows (train, evaluate, etc.)
- Industry-standard pattern used by frameworks like Click, Typer, Argparse