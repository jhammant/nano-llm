import torch
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm  # Progress bar

# ✅ Correct Model Name
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Enable QLoRA (Low-Rank Adapters)
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

# Load the dataset
dataset = load_dataset("json", data_files="nano_docs_clean.jsonl", split="train")

import re

def fix_message_any(msg):
    """
    Convert any `msg` (dict, list, string, whatever) into a dict { "role": "..", "content": ".." }.
    """
    if isinstance(msg, dict):
        return {
            "role": str(msg.get("role", "unknown")),
            "content": str(msg.get("content", "")),
        }
    elif isinstance(msg, list):
        # Heuristic: if length=2, interpret as [role, content]. Otherwise fallback
        if len(msg) == 2:
            return {
                "role": str(msg[0]),
                "content": str(msg[1]),
            }
        else:
            return {
                "role": "unknown",
                "content": str(msg),
            }
    elif isinstance(msg, str):
        # Heuristic: try "role: content" parse
        match = re.match(r"^([^:]+):(.*)$", msg.strip())
        if match:
            return {
                "role": match.group(1).strip(),
                "content": match.group(2).strip(),
            }
        else:
            return {
                "role": "unknown",
                "content": msg,
            }
    else:
        # Fallback for any other type (int, float, None, etc.)
        return {
            "role": "unknown",
            "content": str(msg),
        }


def tokenize_function(example):
    messages = example.get("messages", [])
    fixed_msgs = [fix_message_any(m) for m in messages]

    conversation_parts = [f"{fm['role']}: {fm['content']}" for fm in fixed_msgs]
    conversation = "\n".join(conversation_parts)

    tokenized = tokenizer(
        conversation,
        truncation=True,
        padding="max_length",
        max_length=1024
    )

    # For causal LM training, we set labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=False)

training_args = TrainingArguments(
    output_dir="./nano-llm",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="steps",
    save_total_limit=2,
    bf16=True,  # Use bfloat16 instead of fp16
    report_to="none",
)

# Custom Trainer with Progress Bar
class ProgressTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.pbar = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.state.global_step == 0:  # Start progress bar
            self.start_time = time.time()
            self.pbar = tqdm(total=self.state.max_steps, desc="Training Progress", unit="step")

        loss = super().training_step(model, inputs)

        elapsed_time = time.time() - self.start_time
        steps_completed = self.state.global_step
        steps_remaining = self.state.max_steps - steps_completed
        time_per_step = elapsed_time / steps_completed if steps_completed > 0 else 0
        estimated_time_remaining = steps_remaining * time_per_step

        self.pbar.update(1)
        self.pbar.set_postfix({
            "Loss": round(loss.item(), 4),
            "ETA": f"{estimated_time_remaining:.2f}s"
        })

        # Close progress bar when we've reached total steps
        if steps_completed == self.state.max_steps:
            self.pbar.close()

        return loss

# Trainer
trainer = ProgressTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./nano-llm")
tokenizer.save_pretrained("./nano-llm")
