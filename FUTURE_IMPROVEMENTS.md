# Optional Future Improvements

This document outlines potential enhancements beyond the current refactoring that could further improve code quality and functionality.

---

## Priority 1: High Impact, Low Effort

### 1. Add Type Hints to Conditional Imports
**Current:** Some imports are conditional on arguments
**Suggested:** Use `TYPE_CHECKING` to avoid circular imports if needed

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Trainer
```

### 2. Create a TrainingConfig Dataclass
**Benefit:** Type-safe configuration bundles

```python
from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    model_id: str
    processed_dir: str
    checkpoints_dir: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    logging_steps: int
    save_steps: int
    lr_scheduler: str
    
    @classmethod
    def from_config(cls, config):
        return cls(
            model_id=config.get("model.id"),
            processed_dir=config.get("paths.processed_data_dir"),
            # ... etc
        )
```

### 3. Add Input Validation Functions
**Benefit:** Fail fast with meaningful errors

```python
def validate_dataset_dir(path: str) -> None:
    """Validate that dataset directory exists and is readable."""
    if not os.path.exists(path):
        raise ValueError(f"Dataset directory not found: {path}")
    if not os.path.isdir(path):
        raise ValueError(f"Not a directory: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"Directory not readable: {path}")
```

### 4. Create Enum for Training Method
**Benefit:** Type-safe training method selection

```python
from enum import Enum

class TrainingMethod(Enum):
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"
    QLORA = "qlora"
    PREFIX = "prefix"
```

---

## Priority 2: Code Organization

### 5. Extract Prompt Building Logic to Separate Module
**Before:** Prompt logic mixed with preprocessing
**After:** `data/prompts.py` for all prompt-related logic

```python
# data/prompts.py
class PromptBuilder:
    def __init__(self, system_prompt: str, assistant_marker: str):
        self.system_prompt = system_prompt
        self.assistant_marker = assistant_marker
    
    def build_training_prompt(self, hate: str, cs_type: str, cs: str) -> str:
        ...
    
    def build_inference_prompt(self, hate: str, cs_type: str) -> str:
        ...
```

### 6. Create Base Trainer Class
**Benefit:** Reduce duplication across training modes

```python
# training/base.py
class BaseTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """Override in subclasses"""
        raise NotImplementedError
    
    def prepare_training_args(self) -> TrainingArguments:
        """Override in subclasses"""
        raise NotImplementedError
    
    def train(self):
        """Common training loop"""
        model, tokenizer = self.load_model_and_tokenizer()
        train_dataset, val_dataset = load_dataset(self.config.processed_dir)
        # ... common logic
```

### 7. Create Results Manager Class
**Benefit:** Centralized result handling

```python
class ResultsManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def save_metrics(self, metrics: Dict):
        """Save training metrics"""
    
    def save_model(self, model, tokenizer):
        """Save model and tokenizer"""
    
    def load_best_checkpoint(self):
        """Load best checkpoint"""
```

---

## Priority 3: Testing & CI/CD

### 8. Add Unit Tests
**Suggested Structure:**
```
tests/
├── unit/
│   ├── test_config.py
│   ├── test_prompts.py
│   ├── test_tokenization.py
│   └── test_training_common.py
├── integration/
│   ├── test_preprocessing_pipeline.py
│   └── test_training_pipeline.py
└── fixtures/
    ├── sample_config.yaml
    └── sample_dataset.json
```

### 9. Add Logging Configuration File
**Create:** `logging.ini` or `logging.yaml`

```ini
[loggers]
keys=root,samvad

[handlers]
keys=console,file

[formatters]
keys=standard

[logger_root]
level=WARNING
handlers=console

[logger_samvad]
level=DEBUG
handlers=console,file
qualname=samvad
propagate=0

[handler_console]
class=StreamHandler
level=DEBUG
formatter=standard
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=DEBUG
formatter=standard
args=('training.log',)
```

---

## Priority 4: Documentation

### 10. Add Architecture Documentation
**Create:** `docs/ARCHITECTURE.md`
```markdown
# System Architecture

## Module Dependencies
[Diagram showing module relationships]

## Data Flow
[Sequence diagram for preprocessing → training]

## Configuration Management
[How config flows through the system]
```

### 11. Add API Documentation
**Create:** `docs/API.md` documenting:
- `config.Config` interface
- `training.common` module functions
- `data.preprocess` module functions

### 12. Add Development Guide
**Create:** `docs/DEVELOPMENT.md` with:
- Setting up dev environment
- Running tests
- Code style guidelines
- Making changes

---

## Priority 5: Advanced Features

### 13. Add Configuration Validation Schema
**Using:** `pydantic` for runtime validation

```python
from pydantic import BaseModel, validator

class ConfigSchema(BaseModel):
    model_id: str
    dataset_id: str
    learning_rate: float
    batch_size: int
    
    @validator('learning_rate')
    def learning_rate_range(cls, v):
        if not 1e-5 < v < 1e-1:
            raise ValueError('Learning rate out of reasonable range')
        return v
```

### 14. Add Experiment Tracking
**Using:** `wandb` or `mlflow`

```python
import wandb

def train(output_dir: str = None) -> None:
    wandb.init(project="samvad", name=RUN_NAME)
    
    # ... training code ...
    
    wandb.log({"train_loss": train_loss})
    wandb.finish()
```

### 15. Add Checkpointing with Versioning
**Using:** `hugging_face_hub`

```python
from huggingface_hub import upload_folder

def save_and_upload_checkpoint(model, tokenizer, run_name: str):
    output_dir = f"./checkpoints/{run_name}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Upload to Hub
    upload_folder(
        folder_path=output_dir,
        repo_id="username/samvad-checkpoints",
        repo_type="model",
    )
```

---

## Priority 6: Performance

### 16. Add Profiling Utilities
**Benefit:** Identify bottlenecks

```python
# training/profiling.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{name} took {elapsed:.2f}s")

# Usage
with timer("Dataset loading"):
    train_dataset, val_dataset = load_dataset(PROCESSED_DIR)
```

### 17. Add Memory Monitoring
**Benefit:** Track VRAM usage during training

```python
def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### 18. Add Distributed Training Support
**Using:** `torch.distributed`

```python
def train_distributed(output_dir: str = None) -> None:
    """Support multi-GPU training"""
    # Initialize distributed training
    dist.init_process_group()
    
    model = load_base_model(MODEL_ID)
    model = DDP(model, device_ids=[local_rank])
    
    # ... training code ...
```

---

## Priority 7: Robustness

### 19. Add Graceful Shutdown Handling
**Benefit:** Save state on keyboard interrupt

```python
import signal

def setup_signal_handlers(trainer: Trainer):
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, saving checkpoint...")
        trainer.save_model()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
```

### 20. Add Retry Logic for Network Operations
**Benefit:** Handle transient failures

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def load_dataset_with_retry(dataset_id: str):
    """Load dataset with automatic retry on failure"""
    return load_dataset(dataset_id)
```

---

## Recommended Implementation Order

1. **Phase 1 (Week 1):** Priorities 1-2
   - Quick wins that improve code quality immediately
   - Type safety and configuration management

2. **Phase 2 (Week 2-3):** Priority 3
   - Unit tests and basic CI/CD
   - Ensures code quality is maintainable

3. **Phase 3 (Week 4):** Priorities 4-5
   - Documentation and experiment tracking
   - Makes code professional and reproducible

4. **Phase 4+ (Future):** Priorities 6-7
   - Performance and robustness optimizations
   - As needed based on bottlenecks

---

## Estimated Effort

| Priority | Items | Estimated Time | Difficulty |
|----------|-------|-----------------|------------|
| 1 | 1-4 | 2-3 hours | Low |
| 2 | 5-7 | 4-6 hours | Low-Medium |
| 3 | 8-12 | 6-8 hours | Medium |
| 4 | 13-15 | 8-12 hours | Medium-High |
| 5 | 16-18 | 6-10 hours | Medium |
| 6 | 19-20 | 4-6 hours | Low-Medium |

---

## Quick Wins (Do First!)

If you only have 1-2 hours, implement:
1. TrainingConfig dataclass (Priority 1, Item 2)
2. Input validation functions (Priority 1, Item 3)
3. Basic unit tests (Priority 3, Item 8)

These will provide immediate value with minimal effort.
