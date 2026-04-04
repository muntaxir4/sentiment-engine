from unsloth import FastLanguageModel
import torch
import json
import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from os import path

__DIR__ = path.dirname(__file__)
# --- CONFIGURATION ---
ADAPTER_PATH = path.join(
    __DIR__, "qwen_sentiment_finetuned"
)  # Your local adapter folder
TEST_FILE = path.join(__DIR__, "datasets/test_qwen_28_balanced.jsonl")  # Your test file
MAX_SEQ_LENGTH = 2048

# 1. Load Model (Base + Adapters)
print(f"Loading model from '{ADAPTER_PATH}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

# 2. Define the Template (Must match training exactly)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

# 3. Load & Process Test Data
print(f"Loading test data from {TEST_FILE}...")
test_entries = []
with open(TEST_FILE, "r") as f:
    for line in f:
        if line.strip():
            test_entries.append(json.loads(line))

y_true = []
y_pred = []

print(f"Starting evaluation on {len(test_entries)} examples...")

# 4. Inference Loop
for i, entry in tqdm.tqdm(enumerate(test_entries), total=len(test_entries)):

    # --- A. Parse Ground Truth ---
    # Your file has "output": "{\"emotion\": \"Admiration\", ...}"
    try:
        raw_output = entry["output"]
        ground_truth_json = json.loads(raw_output)  # Parse the inner string
        actual_emotion = ground_truth_json.get("emotion", "Unknown")
    except Exception as e:
        print(f"Skipping bad line {i}: {e}")
        continue

    # --- B. Prepare Prompt ---
    # We use the instruction FROM THE FILE to match training perfectly
    instruction = entry.get("instruction", "Analyze the sentiment...")
    input_text = entry["input"]

    prompt = alpaca_prompt.format(instruction, input_text)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # --- C. Generate ---
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=0.1,
        )

    # --- D. Decode & Parse Prediction ---
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract just the JSON part after "### Response:"
    try:
        answer_part = response_text.split("### Response:")[-1].strip()

        # Clean up potential markdown wrappers
        if "```json" in answer_part:
            answer_part = answer_part.split("```json")[1].split("```")[0].strip()
        elif "```" in answer_part:
            answer_part = answer_part.split("```")[1].strip()

        # Parse prediction
        pred_json = json.loads(answer_part)
        predicted_emotion = pred_json.get("emotion", "Error")
    except:
        predicted_emotion = "ParseError"

    y_true.append(actual_emotion)
    y_pred.append(predicted_emotion)

# 5. Report Results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

# Detailed Report
labels = sorted(list(set(y_true + y_pred)))
print("\n📊 Classification Report:")
# zero_division=0 prevents warnings for emotions that didn't appear in the test set
print(classification_report(y_true, y_pred, zero_division=0))

# 6. Save Confusion Matrix (Optional but impressive)
try:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix (Acc: {acc:.2%})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("evaluation_matrix.png")
    print("\n📸 Saved confusion matrix to 'evaluation_matrix.png'")
except Exception as e:
    print(f"Could not generate plot: {e}")
