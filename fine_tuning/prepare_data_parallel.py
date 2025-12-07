import json
import ollama
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path

__DIR__ = path.dirname(__file__)

# --- CONFIGURATION ---
MODEL_NAME = "qwen2.5:1.5b"
OUTPUT_FILENAME = path.join(__DIR__, "datasets/train_qwen_28_single_parallel.jsonl")
LIMIT = None  # Set to None for full dataset (~43k rows)
MAX_WORKERS = 4  # Parallel threads (Recommended: 4 for RTX 3050)

# --- 1. MAPPINGS ---
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

# --- 2. PRIORITY LOGIC (To pick the "Strongest") ---
priority_order = [
    "grief",
    "remorse",
    "love",
    "hatred",
    "fury",
    "terror",  # Very Intense
    "gratitude",
    "admiration",
    "pride",
    "disgust",
    "embarrassment",  # Strong
    "joy",
    "sadness",
    "anger",
    "fear",
    "excitement",  # Basic
    "annoyance",
    "disapproval",
    "disappointment",
    "confusion",  # Mild
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
    "neutral",  # Always last
]


def pick_best_emotion(labels_indices):
    """
    Given a list of label IDs (e.g. [17, 27]), return the single strongest emotion name.
    """
    current_emotions = [id2label[i] for i in labels_indices]
    # Sort them based on our priority list; earliest in list wins.
    current_emotions.sort(
        key=lambda x: priority_order.index(x) if x in priority_order else 99
    )
    return current_emotions[0]


# --- 3. WORKER FUNCTION ---
def process_row(row):
    """
    Processes a single row in a separate thread.
    """
    text = row["text"]
    labels = row["labels"]

    # A. Pick the Winner
    best_emotion = pick_best_emotion(labels)
    best_emotion_display = best_emotion.capitalize()
    polarity = polarity_map[best_emotion]

    # B. Generate Reasoning (The AI Call)
    prompt = (
        f"Analyze the text and explain in ONE concise sentence why it conveys '{best_emotion}'. "
        f'Text: "{text}"\nReasoning:'
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
        )
        reasoning = response["message"]["content"].strip()
    except Exception:
        reasoning = f"The text contains strong signals of {best_emotion}."

    # C. Format Output
    output_obj = {
        "polarity": polarity,
        "emotion": best_emotion_display,
        "confidence_score": 1.0,
        "reasoning": reasoning,
    }

    entry = {
        "instruction": "Analyze the sentiment. Return JSON with polarity, emotion, confidence_score, and reasoning.",
        "input": text,
        "output": json.dumps(output_obj),
    }

    return [json.dumps(entry)]


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("Downloading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", split="train")

    if LIMIT:
        dataset = dataset.select(range(LIMIT))
        print(f"Limiting to first {LIMIT} rows for testing.")

    print(f"Starting parallel processing with {MAX_WORKERS} workers...")

    final_lines = []

    # Run the ThreadPool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, row) for row in dataset]

        for future in tqdm(as_completed(futures), total=len(dataset)):
            try:
                data = future.result()
                final_lines.extend(data)
            except Exception as e:
                print(f"Error processing row: {e}")

    # Save
    print(f"Saving {len(final_lines)} examples to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, "w") as f:
        for line in final_lines:
            f.write(line + "\n")

    print("Done! Ready for Unsloth training.")
