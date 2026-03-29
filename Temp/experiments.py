
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(tokenizer("Hello, world!"))

dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

print(dataset.column_names)

# for i in range(3):
#     print(dataset[i])

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)

print(dataset["train"].column_names)
print(dataset["test"].column_names)

### where is data collator used in trainer? 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)


model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

training_args = TrainingArguments(
        output_dir= "./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="epoch"
        save_strategy="epoch",,
        load_best_model_at_end=True
)

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer
        )

trainer.train()