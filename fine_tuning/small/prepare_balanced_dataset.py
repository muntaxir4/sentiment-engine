import json
import ollama
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path

__DIR__ = path.dirname(__file__)

# --- CONFIGURATION ---
TARGET_PER_EMOTION = 100  # Aim for 100 examples per emotion (Total ~2800)
MODEL_NAME = "qwen2.5:1.5b"
OUTPUT_FILENAME = path.join(__DIR__, "datasets/train_qwen_28_balanced.jsonl")
MAX_WORKERS = 4

# --- MAPPINGS (Same as before) ---
id2label = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise",
    27: "neutral",
}

polarity_map = {
    "admiration": "Positive",
    "amusement": "Positive",
    "approval": "Positive",
    "caring": "Positive",
    "desire": "Positive",
    "excitement": "Positive",
    "gratitude": "Positive",
    "joy": "Positive",
    "love": "Positive",
    "optimism": "Positive",
    "pride": "Positive",
    "relief": "Positive",
    "anger": "Negative",
    "annoyance": "Negative",
    "disappointment": "Negative",
    "disapproval": "Negative",
    "disgust": "Negative",
    "embarrassment": "Negative",
    "fear": "Negative",
    "grief": "Negative",
    "nervousness": "Negative",
    "remorse": "Negative",
    "sadness": "Negative",
    "confusion": "Neutral",
    "curiosity": "Neutral",
    "realization": "Neutral",
    "surprise": "Neutral",
    "neutral": "Neutral",
}

# Priority to pick single emotion (Same as before)
priority_order = [
    "grief",
    "remorse",
    "love",
    "hatred",
    "fury",
    "terror",
    "gratitude",
    "admiration",
    "pride",
    "disgust",
    "embarrassment",
    "joy",
    "sadness",
    "anger",
    "fear",
    "excitement",
    "annoyance",
    "disapproval",
    "disappointment",
    "confusion",
    "amusement",
    "caring",
    "approval",
    "optimism",
    "relief",
    "realization",
    "curiosity",
    "surprise",
    "desire",
    "nervousness",
    "neutral",
]


def pick_best_emotion(labels_indices):
    current_emotions = [id2label[i] for i in labels_indices]
    current_emotions.sort(
        key=lambda x: priority_order.index(x) if x in priority_order else 99
    )
    return current_emotions[0]


# --- 1. FILTERING & BALANCING LOGIC ---
print("Downloading Data...")
dataset = load_dataset("go_emotions", split="train")

# Buckets for each emotion
buckets = {emotion: [] for emotion in id2label.values()}

print("Sorting data into buckets...")
# Shuffle first to get random variety
shuffled_data = dataset.shuffle(seed=42)

for row in tqdm(shuffled_data):
    # Determine the "winner" emotion for this row
    best_emotion = pick_best_emotion(row["labels"])

    # If we haven't filled this emotion's bucket yet, add it
    if len(buckets[best_emotion]) < TARGET_PER_EMOTION:
        buckets[best_emotion].append(row["text"])

# Flatten the buckets into a single list
selected_rows = []
for emotion, texts in buckets.items():
    print(f"  {emotion}: {len(texts)} examples")
    for text in texts:
        selected_rows.append({"text": text, "emotion": emotion})

print(f"Total selected rows: {len(selected_rows)}")


# --- 2. GENERATION WORKER ---
def process_row(item):
    text = item["text"]
    emotion = item["emotion"]
    emotion_display = emotion.capitalize()
    polarity = polarity_map[emotion]

    # AI Reasoning
    prompt = (
        f"Analyze the text and explain in ONE concise sentence why it conveys '{emotion}'. "
        f'Text: "{text}"\nReasoning:'
    )
    try:
        response = ollama.chat(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
        )
        reasoning = response["message"]["content"].strip()
    except:
        reasoning = f"The text conveys {emotion}."

    # JSON Structure
    output_obj = {
        "polarity": polarity,
        "emotion": emotion_display,
        "confidence_score": 1.0,
        "reasoning": reasoning,
    }

    entry = {
        "instruction": "Analyze the sentiment. Return JSON with polarity, emotion, confidence_score, and reasoning.",
        "input": text,
        "output": json.dumps(output_obj),
    }
    return json.dumps(entry)


# --- 3. PARALLEL EXECUTION ---
print(f"Generating reasoning for {len(selected_rows)} rows...")
final_lines = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for row in selected_rows]
    for future in tqdm(as_completed(futures), total=len(selected_rows)):
        final_lines.append(future.result())

# Save
with open(OUTPUT_FILENAME, "w") as f:
    for line in final_lines:
        f.write(line + "\n")

print("Done! Balanced dataset ready.")
