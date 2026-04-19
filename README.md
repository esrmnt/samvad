# samvad

`samvad` is a small training/evaluation project for counterspeech generation.
It takes hate-speech/counterspeech examples, formats them as chat prompts, and uses Hugging Face tooling to train and compare a few fine-tuning methods.

The current default setup uses

- Model: `Qwen/Qwen2-0.5B-Instruct`
- Dataset: `Aswini123/IntentCONANv2`
- Methods: full fine-tuning, LoRA, QLoRA, and prefix tuning

Most paths, model settings, and training hyperparameters live in `config/config.yaml`.

## Setup

```bash
cd samvad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check that the CLI starts with

```bash
python main.py
```

That command only creates the configured artifact directories and prints the available next steps.

## Workflow

### 1. Preprocess the dataset

Preview a few formatted prompts first

```bash
python main.py --preprocess --model qwen --preview
```

Then build the processed train/validation/test splits

```bash
python main.py --preprocess --model qwen
```

By default this writes to

```text
dataset/processed/{model_slug}/
```

To write somewhere else

```bash
python main.py --preprocess --model qwen --output_dir ./custom_data_path
```

The preprocessing code is in `loaders/preprocess.py`. It can also be imported directly

```python
from loaders.preprocess import preprocess

preprocess()
preprocess(preview=True)
preprocess(output_dir="./my_data")
```

The preprocessing step does the following

1. Loads the configured dataset from Hugging Face.
2. Builds train, validation, and test splits.
3. Converts each row into the chat prompt format used for training.
4. Tokenizes the prompts.
5. Masks labels so loss is only computed on the assistant response.
6. Saves the processed dataset to disk.

### 2. Train a model

Run one method at a time

```bash
# Full fine-tuning
python main.py --train --method full --model qwen

# LoRA
python main.py --train --method lora --model qwen

# QLoRA
python main.py --train --method qlora --model qwen

# Prefix tuning
python main.py --train --method prefix --model qwen
```

Checkpoints are saved under

```text
dataset/checkpoints/{model_slug}/{method}/
```

Training resumes from the most recent checkpoint in the method directory when one is available.

### 3. Generate outputs

After training, generate counterspeech for the held-out test split

```bash
python main.py --generate --model qwen
```

The generation script looks for available checkpoints and writes CSV files to

```text
dataset/results/{model_slug}/generated/
```

Each generated CSV contains

```text
hate_speech, cs_type, reference, generated
```

### 4. Evaluate generated outputs

```bash
python main.py --evaluate --model qwen
```

Evaluation reads the generated CSV files and writes

```text
dataset/results/{model_slug}/metrics/scores.csv
dataset/results/{model_slug}/metrics/scores.json
```

The current evaluation script reports overlap, readability, embedding, toxicity, novelty, and diversity metrics, including BLEU, ROUGE, METEOR, GLEU, CoSIM, BERTScore, Detoxify toxicity, and distinct-2 diversity.

## Results

The table below is from the current Qwen run on the held-out test split. These
numbers are useful for comparing the runs in this repo, but they should not be
read as a complete judgment of counterspeech quality.

Full outputs are saved in

```text
dataset/results/qwen2-0/metrics/scores.csv
dataset/results/qwen2-0/metrics/scores.json
```

Main comparison

| Method | BLEU | ROUGE-L | METEOR | CoSIM | BERT Score | Toxicity | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full fine-tuning | 0.0582 | 0.1809 | 0.1790 | 0.4770 | 0.2685 | 0.0056 | 0.1707 |
| LoRA | 0.0607 | 0.1851 | 0.1850 | 0.5164 | 0.2858 | 0.0096 | 0.2836 |
| QLoRA | 0.0557 | 0.1812 | 0.1839 | 0.5094 | 0.2807 | 0.0120 | 0.2847 |
| Prefix tuning | 0.0352 | 0.1593 | 0.1707 | 0.4872 | 0.2550 | 0.0080 | 0.2771 |

Additional metrics

| Method | ROUGE-1 | ROUGE-2 | GLEU | Repetition Rate | Flesch Reading Ease | Novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full fine-tuning | 0.2593 | 0.0729 | 0.0682 | 0.0023 | 39.33 | 0.9919 |
| LoRA | 0.2731 | 0.0745 | 0.0705 | 0.0003 | 36.38 | 0.9908 |
| QLoRA | 0.2689 | 0.0708 | 0.0676 | 0.0003 | 37.37 | 0.9912 |
| Prefix tuning | 0.2459 | 0.0541 | 0.0547 | 0.0002 | 38.24 | 0.9888 |

## Configuration

The main config file is

```text
config/config.yaml
```

Useful sections

- `model` selects the active model alias.
- `models` stores model aliases, Hugging Face IDs, and folder slugs.
- `dataset` stores the Hugging Face dataset ID.
- `data` controls max sequence length, split sizes, and random seed.
- `paths` controls where processed data, checkpoints, and results are stored.
- `prompts` stores the system prompt and assistant marker.
- `training` stores shared training hyperparameters.
- `lora`, `qlora`, and `prefix` store method-specific settings.
- `generation` stores inference settings.

Config values are available from Python with dot notation

```python
from config import config

model_id = config.get("model.id")
batch_size = config.get("training.batch_size")
learning_rate = config.get("training.learning_rate", default=2e-4)
```

To add another model, add a new alias under `models` and set `model.name` to that alias. Before venturing into this path however, please see the section "**Future Works**"

```yaml
model:
  name: "qwen"

models:
  qwen:
    id: "Qwen/Qwen2-0.5B-Instruct"
    slug: "qwen2-0"
```

## Notes

- The training scripts use Hugging Face `Trainer`.
- LoRA, QLoRA, and prefix tuning use `peft`.
- QLoRA depends on `bitsandbytes`.
- The current code is mainly set up around Qwen-style causal chat models. Other decoder-only models may work with config changes, but prompt formatting, tokenizer behavior and LoRA target modules should be checked before training.

## Future Work

One useful next step is to make model support more generic. Ideally, adding a new model should only require editing `config/config.yaml`. To get there, the code should move model-specific pieces into config or small adapter helpers, especially

- chat prompt formatting, preferably using each tokenizer's chat template when available
- tokenizer padding and special-token handling
- LoRA/QLoRA target module names
- causal-LM versus encoder-decoder model loading
- offline/cache behavior for newly added models

## License

This project is licensed under LGPL-2.1. See `LICENSE` for the full text.
