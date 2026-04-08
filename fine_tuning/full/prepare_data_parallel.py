import json
import re
import os
import ollama
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from os import path

try:
    __DIR__ = path.dirname(path.abspath(__file__))
except NameError:
    __DIR__ = os.getcwd()

# --- CONFIGURATION ---
MODEL_NAME = "qwen2.5:1.5b"
OUTPUT_FILENAME = path.join(__DIR__, "datasets/train_qwen_28_full.jsonl")
LIMIT = None  # Set to None for full dataset (~43k rows)
MAX_WORKERS = 4  # Parallel threads (Recommended: 4 for RTX 3050)
MAX_IN_FLIGHT = MAX_WORKERS * 4
RESUME = True
CHECKPOINT_EVERY = 100

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


def aggregate_polarity(emotions):
    polarities = {polarity_map.get(emotion, "Neutral") for emotion in emotions}
    return polarities.pop() if len(polarities) == 1 else "Mixed"


def _extract_json_object(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def build_default_emotions(label_emotions):
    if not label_emotions:
        return [{"emotion": "neutral", "confidence_score": 1.0}]

    score = round(1.0 / len(label_emotions), 4)
    confs = []
    running_total = 0.0
    for i, emotion in enumerate(label_emotions):
        if i == len(label_emotions) - 1:
            value = round(1.0 - running_total, 4)
        else:
            value = score
            running_total += value
        confs.append({"emotion": emotion, "confidence_score": max(0.0, min(1.0, value))})
    return confs


def normalize_model_emotions(model_data, label_emotions):
    if not isinstance(model_data, dict):
        return build_default_emotions(label_emotions), "Model output parsing failed."

    raw_emotions = model_data.get("emotions", [])
    if not isinstance(raw_emotions, list):
        raw_emotions = []

    label_set = set(label_emotions)
    normalized = []
    used = set()
    for item in raw_emotions:
        if not isinstance(item, dict):
            continue

        raw_name = str(item.get("emotion", "")).strip().lower()
        if raw_name not in label_set or raw_name in used:
            continue

        try:
            raw_score = float(item.get("confidence_score", 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0

        score = max(0.0, min(1.0, raw_score))
        normalized.append({"emotion": raw_name, "confidence_score": score})
        used.add(raw_name)

    if not normalized:
        normalized = build_default_emotions(label_emotions)

    total = sum(item["confidence_score"] for item in normalized)
    if total <= 0:
        normalized = build_default_emotions([item["emotion"] for item in normalized])
    else:
        for item in normalized:
            item["confidence_score"] = round(item["confidence_score"] / total, 4)

    missing = [emotion for emotion in label_emotions if emotion not in used]
    if missing:
        share = round(0.1 / len(missing), 4)
        for emotion in missing:
            normalized.append({"emotion": emotion, "confidence_score": share})

        total = sum(item["confidence_score"] for item in normalized)
        for item in normalized:
            item["confidence_score"] = round(item["confidence_score"] / total, 4)

    reasoning = str(model_data.get("reasoning", "")).strip()
    if not reasoning:
        reasoning = "The text expresses multiple emotions with varying intensity."

    return normalized, reasoning


def load_processed_row_ids(output_file):
    processed = set()
    if not path.exists(output_file):
        return processed

    missing_row_id_count = 0
    with open(output_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                row_id = entry.get("meta", {}).get("row_id")
                if isinstance(row_id, int):
                    processed.add(row_id)
                else:
                    missing_row_id_count += 1
            except json.JSONDecodeError:
                continue

    if missing_row_id_count:
        print(
            f"Warning: Found {missing_row_id_count} existing lines without meta.row_id. "
            "Resume will only skip lines that contain row_id."
        )

    return processed


# --- 3. WORKER FUNCTION ---
def process_row(row_id, row):
    """
    Processes a single row in a separate thread.
    """
    text = row["text"]
    labels = row["labels"]
    label_emotions = [id2label[i] for i in labels]

    # A. Candidate emotions + aggregate polarity
    polarity = aggregate_polarity(label_emotions)

    # B. Generate Reasoning (The AI Call)
    prompt = (
        "You are a sentiment labeling assistant. "
        "Use ONLY the provided candidate emotions. "
        "Return STRICT JSON with this schema: "
        '{"emotions":[{"emotion":"<candidate>","confidence_score":0.0}],"reasoning":"..."}. '
        "Rules: confidence_score must be between 0 and 1, include one item per candidate emotion, and scores must sum to 1. "
        f"Candidate emotions: {label_emotions}. "
        f'Text: "{text}"'
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
        )
        raw_content = response["message"]["content"].strip()
        parsed = _extract_json_object(raw_content)
        emotions_conf, reasoning = normalize_model_emotions(parsed, label_emotions)
    except Exception:
        emotions_conf = build_default_emotions(label_emotions)
        reasoning = "The text expresses multiple emotions with varying intensity."

    # C. Format Output
    output_obj = {
        "polarity": polarity,
        "emotions": [
            {
                "emotion": item["emotion"].capitalize(),
                "confidence_score": item["confidence_score"],
            }
            for item in emotions_conf
        ],
        "reasoning": reasoning,
    }

    entry = {
        "instruction": "Analyze the sentiment. Return JSON with polarity, emotions (with confidence_score), and reasoning.",
        "input": text,
        "output": json.dumps(output_obj),
        "meta": {"row_id": row_id},
    }

    return json.dumps(entry)


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("Downloading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", split="train")

    if LIMIT:
        dataset = dataset.select(range(LIMIT))
        print(f"Limiting to first {LIMIT} rows for testing.")

    total_rows = len(dataset)
    processed_row_ids = load_processed_row_ids(OUTPUT_FILENAME) if RESUME else set()

    pending_row_ids = [
        row_id for row_id in range(total_rows) if row_id not in processed_row_ids
    ]

    print(
        f"Starting parallel processing with {MAX_WORKERS} workers "
        f"(in-flight: {MAX_IN_FLIGHT})."
    )
    print(
        f"Resume mode: {'ON' if RESUME else 'OFF'} | "
        f"already processed: {len(processed_row_ids)} | pending: {len(pending_row_ids)}"
    )

    if not pending_row_ids:
        print("No pending rows. Dataset output is already complete.")
        raise SystemExit(0)

    write_mode = "a" if RESUME else "w"
    rows_since_flush = 0
    completed = 0

    def submit_next(executor, futures_dict, pending_iter):
        try:
            row_id = next(pending_iter)
        except StopIteration:
            return False

        future = executor.submit(process_row, row_id, dataset[row_id])
        futures_dict[future] = row_id
        return True

    # Run the ThreadPool with bounded in-flight tasks
    with open(OUTPUT_FILENAME, write_mode) as out_handle, ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        pending_iter = iter(pending_row_ids)
        futures = {}

        for _ in range(min(MAX_IN_FLIGHT, len(pending_row_ids))):
            submit_next(executor, futures, pending_iter)

        with tqdm(total=len(pending_row_ids)) as progress_bar:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    row_id = futures.pop(future)
                    try:
                        line = future.result()
                        out_handle.write(line + "\n")
                    except Exception as e:
                        print(f"Error processing row_id {row_id}: {e}")

                    completed += 1
                    rows_since_flush += 1
                    progress_bar.update(1)

                    if rows_since_flush >= CHECKPOINT_EVERY:
                        out_handle.flush()
                        rows_since_flush = 0

                    submit_next(executor, futures, pending_iter)

        out_handle.flush()

    print(f"Saved {completed} new examples to {OUTPUT_FILENAME}.")

    print("Done! Ready for Unsloth training.")
