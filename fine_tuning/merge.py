from unsloth import FastLanguageModel
from peft import PeftModel
import os
import shutil
from os import path

__DIR__ = path.dirname(__file__)

# --- CONFIGURATION ---
ADAPTER_DIR = path.join(__DIR__, "qwen_sentiment_finetuned")
MERGED_DIR = path.join(__DIR__, "qwen_merged_16bit")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. Load Base Model (Must be 16-bit for merging)
print(f"Loading base model: {BASE_MODEL}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,  # CRITICAL: False for merging
)

# 2. Load Adapters
print(f"Loading adapters from {ADAPTER_DIR}...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

# 3. Merge
print("Merging weights...")
model = model.merge_and_unload()

# 4. Save Full Model
print(f"Saving to {MERGED_DIR}...")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

# 5. Cleanup Index File (Fixes the "0 tensors" bug)
index_file = os.path.join(MERGED_DIR, "model.safetensors.index.json")
if os.path.exists(index_file):
    print("Removing unnecessary index file...")
    os.remove(index_file)

print("✅ Merge complete! Ready for Docker.")
