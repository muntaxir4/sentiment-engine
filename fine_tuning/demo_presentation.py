#!/usr/bin/env python3
"""
Presentation demo: shows the data refinement (28 -> 6 emotions) and the
improved evaluation metrics on a curated 100-sample subset.

Usage:
    python demo_presentation.py                       # full demo (calls Ollama)
    python demo_presentation.py --skip-eval           # only show data processing
    python demo_presentation.py --model sentiment-engine-full
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from os import path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from full.evaluate_ollama import (
    build_prompt,
    extract_first_json_object,
    extract_primary_emotion,
    parse_json_maybe,
    query_ollama,
)

__DIR__ = path.dirname(path.abspath(__file__))
SOURCE_FILE = path.join(__DIR__, "small/datasets/test_qwen_28_balanced.jsonl")
CURATED_FILE = path.join(__DIR__, "demo_eval_6class_100.jsonl")

CORE_EMOTIONS = ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"]
SAMPLES_PER_CLASS = 100 // len(CORE_EMOTIONS)  # 16, topped up to 100

BAR = "=" * 64
SUB = "-" * 64


def section(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


def pause(seconds=1.2):
    time.sleep(seconds)


def load_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_of(entry):
    gt = parse_json_maybe(entry.get("output", entry.get("expected_output")))
    return extract_primary_emotion(gt)


def demo_data_processing(all_rows):
    print("\n\n")
    section("STEP 1  ·  DATA PROCESSING")

    raw_counts = Counter(label_of(r) for r in all_rows)
    print(f"\nRaw GoEmotions test split   : {len(all_rows)} samples across "
          f"{len(raw_counts)} classes")
    print("Sample of the 28 raw classes :",
          ", ".join(sorted(raw_counts)[:8]) + ", ...")

    print(f"\n{SUB}\nBEFORE  ·  raw 28-class record")
    print(SUB)
    sample_raw = next(r for r in all_rows if label_of(r) not in CORE_EMOTIONS)
    print(json.dumps(
        {
            "input": sample_raw["input"],
            "output": parse_json_maybe(sample_raw["output"]),
            "_label_space": f"{len(raw_counts)} emotions",
        },
        indent=2,
    ))
    pause()

    print(f"\n{SUB}\nAFTER   ·  filtered + normalized to 6 core emotions")
    print(SUB)
    sample_clean = next(r for r in all_rows if label_of(r) in CORE_EMOTIONS)
    cleaned = parse_json_maybe(sample_clean["output"])
    cleaned_view = {
        "input": sample_clean["input"],
        "output": {
            "polarity": cleaned.get("polarity"),
            "emotion": cleaned.get("emotion"),
            "confidence_score": cleaned.get("confidence_score"),
        },
        "_label_space": f"{len(CORE_EMOTIONS)} core emotions: {CORE_EMOTIONS}",
    }
    print(json.dumps(cleaned_view, indent=2))
    pause()


def build_curated_eval(all_rows, seed=7):
    section("STEP 2  ·  CURATED EVALUATION SET  (100 samples · 6 emotions)")

    if path.exists(CURATED_FILE):
        curated = load_jsonl(CURATED_FILE)
        print(f"\nReusing existing curated set ← {CURATED_FILE}")
    else:
        by_label = defaultdict(list)
        for r in all_rows:
            lbl = label_of(r)
            if lbl in CORE_EMOTIONS:
                by_label[lbl].append(r)

        rng = random.Random(seed)
        curated = []
        for lbl in CORE_EMOTIONS:
            pool = by_label.get(lbl, [])
            rng.shuffle(pool)
            curated.extend(pool[:SAMPLES_PER_CLASS])

        leftovers = [r for lbl in CORE_EMOTIONS
                     for r in by_label[lbl][SAMPLES_PER_CLASS:]]
        rng.shuffle(leftovers)
        curated.extend(leftovers[: 100 - len(curated)])
        curated = curated[:100]

        with open(CURATED_FILE, "w", encoding="utf-8") as f:
            for entry in curated:
                f.write(json.dumps(entry) + "\n")
        print(f"\nWrote {len(curated)} samples → {CURATED_FILE}")

    dist = Counter(label_of(r) for r in curated)
    print("\nClass distribution:")
    for lbl in CORE_EMOTIONS:
        print(f"  {lbl:<10s} {dist.get(lbl, 0):>3d}")
    pause()
    return curated


def evaluate(curated, model, ollama_url, timeout):
    section(f"STEP 3  ·  EVALUATION  (model: {model})")
    print(f"\nQuerying Ollama on {len(curated)} samples...\n")

    try:
        import tqdm
        iterator = tqdm.tqdm(curated, ncols=72)
    except ImportError:
        iterator = curated

    y_true, y_pred = [], []
    parse_errors = request_errors = 0

    for entry in iterator:
        actual = label_of(entry)
        prompt = build_prompt(entry)
        try:
            raw = query_ollama(ollama_url, model, prompt, timeout)
        except Exception as e:
            request_errors += 1
            y_true.append(actual)
            y_pred.append("RequestError")
            print(f"  ! request error: {e}")
            continue
        try:
            obj = parse_json_maybe(raw) or extract_first_json_object(raw)
            predicted = extract_primary_emotion(obj)
            if predicted not in CORE_EMOTIONS:
                predicted = "Other"
        except Exception:
            parse_errors += 1
            predicted = "ParseError"
        y_true.append(actual)
        y_pred.append(predicted)

    return y_true, y_pred, parse_errors, request_errors


def report(y_true, y_pred, parse_errors, request_errors):
    section("STEP 4  ·  RESULTS")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    def row(name, value):
        marker = "PASS" if value > 0.50 else "----"
        print(f"  {name:<11s} {value:6.4f}   [{marker}]")

    print()
    row("Accuracy",  acc)
    row("Precision", prec)
    row("Recall",    rec)
    row("F1-score",  f1)
    print(f"\n  parse errors  : {parse_errors}/{len(y_true)}")
    print(f"  request errors: {request_errors}/{len(y_true)}")

    print(f"\n{SUB}\nPer-class report\n{SUB}")
    print(classification_report(
        y_true, y_pred,
        labels=CORE_EMOTIONS,
        zero_division=0,
    ))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sentiment-engine-full")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    p.add_argument("--timeout", type=float, default=120.0)
    return p.parse_args()


def main():
    args = parse_args()

    section("FINAL YEAR PROJECT  ·  EMOTION CLASSIFICATION DEMO")
    pause()

    all_rows = load_jsonl(SOURCE_FILE)
    demo_data_processing(all_rows)
    curated = build_curated_eval(all_rows)


    y_true, y_pred, pe, re_ = evaluate(
        curated, args.model, args.ollama_url, args.timeout
    )
    report(y_true, y_pred, pe, re_)
    print()


if __name__ == "__main__":
    main()
