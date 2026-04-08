from unsloth import FastLanguageModel
import torch
import json
import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
import re

__DIR__ = path.dirname(__file__)
# --- CONFIGURATION ---
ADAPTER_PATH = path.join(
    __DIR__, "qwen_sentiment_finetuned_full"
)  # Full adapter folder
TEST_FILE = path.join(__DIR__, "datasets/test_qwen_28_full.jsonl")  # Full test file
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
parse_errors = 0


def _parse_json_maybe(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _normalize_emotion_label(label):
    if not isinstance(label, str):
        return "Unknown"
    label = label.strip()
    return label.capitalize() if label else "Unknown"


def _extract_primary_emotion(obj):
    # New schema: emotions = [{"emotion": "...", "confidence_score": 0.9}, ...]
    emotions = obj.get("emotions")
    if isinstance(emotions, list) and emotions:
        best_label = None
        best_conf = float("-inf")
        for item in emotions:
            if isinstance(item, dict):
                label = _normalize_emotion_label(item.get("emotion", "Unknown"))
                conf = item.get("confidence_score", 0.0)
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0
                if conf > best_conf and label != "Unknown":
                    best_conf = conf
                    best_label = label
            elif isinstance(item, str):
                label = _normalize_emotion_label(item)
                if label != "Unknown" and best_label is None:
                    best_label = label
        if best_label:
            return best_label

    # Backward-compatible fallback
    if "emotion" in obj:
        return _normalize_emotion_label(obj.get("emotion", "Unknown"))

    return "Unknown"


def _extract_first_json_object(text):
    # Handles cases where the model emits partial/repeated JSON.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in response")
    candidate = match.group(0)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(candidate)
    return parsed

print(f"Starting evaluation on {len(test_entries)} examples...")

# 4. Inference Loop
for i, entry in tqdm.tqdm(enumerate(test_entries), total=len(test_entries)):

    # --- A. Parse Ground Truth ---
    try:
        raw_output = entry.get("output", entry.get("expected_output"))
        ground_truth_json = _parse_json_maybe(raw_output)
        actual_emotion = _extract_primary_emotion(ground_truth_json)
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
            temperature=0.0,
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

        pred_json = _extract_first_json_object(answer_part)
        predicted_emotion = _extract_primary_emotion(pred_json)
    except:
        predicted_emotion = "ParseError"
        parse_errors += 1

    y_true.append(actual_emotion)
    y_pred.append(predicted_emotion)

# 5. Report Results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {acc:.2%}")
print(f"⚠️ Parse errors: {parse_errors}/{len(y_true)} ({(parse_errors / max(len(y_true), 1)):.2%})")

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
    cm_out = path.join(__DIR__, "evaluation_matrix_full.png")
    plt.savefig(cm_out)
    print(f"\n📸 Saved confusion matrix to '{cm_out}'")
except Exception as e:
    print(f"Could not generate plot: {e}")
