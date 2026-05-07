from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# Configuration
max_seq_length = 2048
load_in_4bit = True
DATASET_PATH = "../data/traffic_law_dataset.jsonl"
OUTPUT_DIR = "../models/phi3-traffic-lora"

print("1. Loading Base Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=load_in_4bit,
)

print("2. Configuring LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("3. Formatting Dataset...")
prompt_template = """<|system|>
{instruction}<|end|>
<|user|>
Claim: {input}<|end|>
<|assistant|>
{output}<|end|>"""

EOS_TOKEN = tokenizer.eos_token

def formatting_func(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        text = prompt_template.format(instruction=inst, input=inp, output=out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_func, batched=True)

print("4. Starting SFT Training...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,         # Better for industry datasets
        num_train_epochs=3,       # Train by epochs, not max_steps
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="training_checkpoints",
    ),
)

trainer.train()

print(f"5. Saving Fine-Tuned Adapters to {OUTPUT_DIR}...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✓ Fine-tuning complete!")