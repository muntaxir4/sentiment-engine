#!/usr/bin/env python3
import argparse
import json
import re
from os import path
from collections import Counter

import matplotlib.pyplot as plt
import requests
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tqdm

__DIR__ = path.dirname(__file__)
DEFAULT_TEST_FILE = path.join(__DIR__, "datasets/test_qwen_28_full.jsonl")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "sentiment-engine-full"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate full sentiment model via Ollama.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Ollama model name, e.g. sentiment-full")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="JSONL test file path")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama generate API URL")
    parser.add_argument(
        "--out-cm",
        default=path.join(__DIR__, "evaluation_matrix_full_ollama.png"),
        help="Output path for confusion matrix image",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    return parser.parse_args()


def parse_json_maybe(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def normalize_label(label):
    if not isinstance(label, str):
        return "Unknown"
    label = label.strip()
    return label.capitalize() if label else "Unknown"


def extract_primary_emotion(obj):
    emotions = obj.get("emotions")
    if isinstance(emotions, list) and emotions:
        best_label = None
        best_conf = float("-inf")
        for item in emotions:
            if isinstance(item, dict):
                label = normalize_label(item.get("emotion", "Unknown"))
                conf = item.get("confidence_score", 0.0)
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0
                if label != "Unknown" and conf > best_conf:
                    best_conf = conf
                    best_label = label
            elif isinstance(item, str) and best_label is None:
                label = normalize_label(item)
                if label != "Unknown":
                    best_label = label
        if best_label:
            return best_label

    if "emotion" in obj:
        return normalize_label(obj.get("emotion", "Unknown"))
    return "Unknown"


def extract_first_json_object(text):
    # Prefer raw decoder to tolerate trailing tokens.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON object found")


def build_prompt(entry):
    instruction = entry.get(
        "instruction",
        "Analyze the sentiment. Return JSON with polarity, emotions (with confidence_score), and reasoning.",
    )
    input_text = entry.get("input", "")
    return f"{instruction}\nInput: {input_text}"


def query_ollama(ollama_url, model, prompt, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }
    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    return body.get("response", "")


def main():
    args = parse_args()

    print(f"Loading test data from {args.test_file}...")
    test_entries = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_entries.append(json.loads(line))

    print(f"Starting Ollama evaluation on {len(test_entries)} examples...")
    y_true = []
    y_pred = []
    parse_errors = 0
    request_errors = 0

    for i, entry in tqdm.tqdm(enumerate(test_entries), total=len(test_entries)):
        try:
            raw_output = entry.get("output", entry.get("expected_output"))
            gt_obj = parse_json_maybe(raw_output)
            actual_emotion = extract_primary_emotion(gt_obj)
        except Exception as e:
            print(f"Skipping bad GT line {i}: {e}")
            continue

        prompt = build_prompt(entry)
        try:
            raw_response = query_ollama(args.ollama_url, args.model, prompt, args.timeout)
        except Exception as e:
            request_errors += 1
            y_true.append(actual_emotion)
            y_pred.append("RequestError")
            print(f"Request error at line {i}: {e}")
            continue

        try:
            pred_obj = parse_json_maybe(raw_response)
            if not pred_obj:
                pred_obj = extract_first_json_object(raw_response)
            predicted_emotion = extract_primary_emotion(pred_obj)
        except Exception:
            parse_errors += 1
            predicted_emotion = "ParseError"

        y_true.append(actual_emotion)
        y_pred.append(predicted_emotion)

    print("\n" + "=" * 50)
    print("FINAL RESULTS (OLLAMA)")
    print("=" * 50)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print(f"Parse errors: {parse_errors}/{len(y_true)} ({(parse_errors / max(len(y_true), 1)):.2%})")
    print(f"Request errors: {request_errors}/{len(y_true)} ({(request_errors / max(len(y_true), 1)):.2%})")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nData Diagnostics:")
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    print(f"- Unique true labels: {len(true_counts)}")
    print(f"- Unique predicted labels: {len(pred_counts)}")

    print("\nClass Support (true -> predicted):")
    for label in sorted(set(true_counts) | set(pred_counts)):
        print(f"- {label}: true={true_counts.get(label, 0)}, pred={pred_counts.get(label, 0)}")

    # Top confusion pairs excluding correct predictions
    pair_counts = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            pair_counts[(t, p)] += 1

    print("\nTop Confusions (true -> pred):")
    if pair_counts:
        for (t, p), n in pair_counts.most_common(15):
            print(f"- {t} -> {p}: {n}")
    else:
        print("- None")

    try:
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix (Ollama, Acc: {acc:.2%})")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(args.out_cm)
        print(f"\nSaved confusion matrix to '{args.out_cm}'")
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")


if __name__ == "__main__":
    main()
