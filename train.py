import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# =========================
# Load config
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# =========================
# QLoRA quantization
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg["load_in_4bit"],
    bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"])
)

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Model
# =========================
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name_or_path"],
    quantization_config=bnb_config,
    device_map="auto"
)

if cfg["gradient_checkpointing"]:
    model.gradient_checkpointing_enable()

# =========================
# LoRA
# =========================
lora_config = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    target_modules=cfg["target_modules"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# Dataset
# =========================
dataset = load_dataset(cfg["dataset_path"])

def tokenize_fn(example):
    return tokenizer(
        example[cfg["dataset_text_field"]],
        truncation=True,
        max_length=cfg["max_seq_length"]
    )

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["num_train_epochs"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    logging_steps=cfg["logging_steps"],
    save_steps=cfg["save_steps"],
    fp16=cfg["fp16"],
    bf16=cfg["bf16"],
    optim="paged_adamw_8bit",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer
)

trainer.train()
