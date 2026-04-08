import json
import re
import os
import time
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from os import path

# --- CONFIGURATION ---
# Using the HF model name directly for vLLM
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_FILENAME = "/content/drive/MyDrive/sentiment-engine/datasets/train_qwen_28_full.jsonl"
BATCH_SIZE = 500  # Adjust based on VRAM (T4 handles ~500-1000 for 1.5B)
RESUME = True

# --- 1. MAPPINGS (Consistent with prepare_data_parallel.py) ---
id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral",
}

polarity_map = {
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive",
    "caring": "Positive", "desire": "Positive", "excitement": "Positive",
    "gratitude": "Positive", "joy": "Positive", "love": "Positive",
    "optimism": "Positive", "pride": "Positive", "relief": "Positive",
    "anger": "Negative", "annoyance": "Negative", "disappointment": "Negative",
    "disapproval": "Negative", "disgust": "Negative", "embarrassment": "Negative",
    "fear": "Negative", "grief": "Negative", "nervousness": "Negative",
    "remorse": "Negative", "sadness": "Negative", "confusion": "Neutral",
    "curiosity": "Neutral", "realization": "Neutral", "surprise": "Neutral",
    "neutral": "Neutral",
}

def aggregate_polarity(emotions):
    polarities = {polarity_map.get(emotion, "Neutral") for emotion in emotions}
    return polarities.pop() if len(polarities) == 1 else "Mixed"

def _extract_json_object(raw_text):
    # Try to extract the first valid JSON object from the text
    # 1. Try direct load
    try:
        return json.loads(raw_text)
    except Exception:
        pass
    # 2. Try fenced code block
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except Exception:
            pass
    # 3. Try all { ... } blocks, pick first that parses
    matches = re.findall(r'\{.*?\}', raw_text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except Exception:
            continue
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
        if not isinstance(item, dict): continue
        raw_name = str(item.get("emotion", "")).strip().lower()
        if raw_name not in label_set or raw_name in used: continue
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

    reasoning = str(model_data.get("reasoning", "")).strip()
    if not reasoning:
        if len(normalized) == 1:
            reasoning = f"The text expresses the emotion '{normalized[0]['emotion']}'."
        else:
            emotion_list = ', '.join([item['emotion'] for item in normalized])
            reasoning = f"The text expresses multiple emotions: {emotion_list}."
    return normalized, reasoning

def load_processed_row_ids(output_file):
    processed = set()
    if not path.exists(output_file):
        return processed
    with open(output_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                row_id = entry.get("meta", {}).get("row_id")
                if isinstance(row_id, int):
                    processed.add(row_id)
            except json.JSONDecodeError:
                continue
    return processed

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", split="train")
    
    processed_row_ids = load_processed_row_ids(OUTPUT_FILENAME) if RESUME else set()
    pending_indices = [i for i in range(len(dataset)) if i not in processed_row_ids]
    
    print(f"Total rows: {len(dataset)} | Already processed: {len(processed_row_ids)} | Pending: {len(pending_indices)}")
    
    if not pending_indices:
        print("All rows already processed.")
        exit(0)

    # Initialize vLLM
    # T4 has 16GB, 1.5B model fits easily.
    llm = LLM(model=MODEL_NAME, trust_remote_code=True, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

    write_mode = "a" if RESUME else "w"
    
    with open(OUTPUT_FILENAME, write_mode) as out_handle:
        for i in tqdm(range(0, len(pending_indices), BATCH_SIZE)):
            batch_indices = pending_indices[i : i + BATCH_SIZE]
            prompts = []
            batch_metadata = []

            for idx in batch_indices:
                row = dataset[idx]
                text = row["text"]
                labels = row["labels"]
                label_emotions = [id2label[l] for l in labels]
                prompt_text = (
                    "You are a sentiment labeling assistant. "
                    "Use ONLY the provided candidate emotions. "
                    "Return STRICT JSON with this schema: "
                    '{"emotions":[{"emotion":"<candidate>","confidence_score":0.0}],"reasoning":"..."}'
                    " ALWAYS include a 'reasoning' field in your JSON output that explains your choice(s). "
                    f"Candidate emotions: {label_emotions}. "
                    f'Text: "{text}"'
                )
                # Format using Qwen 2.5 Chat template
                full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
                prompts.append(full_prompt)
                batch_metadata.append({"row_id": idx, "label_emotions": label_emotions, "text": text})

            # Batch Generate
            outputs = llm.generate(prompts, sampling_params)

            for output, meta in zip(outputs, batch_metadata):
                raw_content = output.outputs[0].text.strip()
                parsed = _extract_json_object(raw_content)
                emotions_conf, reasoning = normalize_model_emotions(parsed, meta["label_emotions"])
                
                polarity = aggregate_polarity(meta["label_emotions"])
                
                output_obj = {
                    "polarity": polarity,
                    "emotions": [
                        {"emotion": item["emotion"].capitalize(), "confidence_score": item["confidence_score"]}
                        for item in emotions_conf
                    ],
                    "reasoning": reasoning,
                }

                entry = {
                    "instruction": "Analyze the sentiment. Return JSON with polarity, emotions (with confidence_score), and reasoning.",
                    "input": meta["text"],
                    "output": json.dumps(output_obj),
                    "meta": {"row_id": meta["row_id"]},
                }
                out_handle.write(json.dumps(entry) + "\n")
            
            out_handle.flush()
            os.fsync(out_handle.fileno())

    print(f"Done! Processed {len(pending_indices)} rows.")
