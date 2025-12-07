from unsloth import FastLanguageModel
import torch
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from datasets import load_dataset
from os import path

__DIR__ = path.dirname(__file__)

# --- CONFIGURATION ---
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = False
DATASET_FILE = path.join(__DIR__, "datasets/train_qwen_28_balanced.jsonl")
OUTPUT_DIR = path.join(__DIR__, "qwen_sentiment_finetuned")

# 1. Load Model & Tokenizer
print("Loading Qwen 2.5 1.5B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 3. Format the Data
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


print(f"Loading dataset: {DATASET_FILE}")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Set up Trainer (Optimized for TRL 0.25.1)
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    # --- ARGS MOVED TO CONFIG IN TRL 0.25+ ---
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    dataset_num_proc=2,
    packing=False,
    # -----------------------------------------
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    # max_steps=60,
    num_train_epochs=1,  # Change to num_train_epochs=1 for the full run!
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # New name for 'tokenizer'
    train_dataset=dataset,
    args=sft_config,
)

# 5. Train
print("Starting Training...")
trainer_stats = trainer.train()

# 6. Save Adapters Only (SAFE)
print("Training complete. Saving adapters...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Done! Adapters saved in '{OUTPUT_DIR}'.")
print("Now run 'python merge.py' followed by the Docker command.")
