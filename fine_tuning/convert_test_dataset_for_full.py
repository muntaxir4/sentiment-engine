#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def safe_parse_output(output_field):
    # old rows store JSON as a string inside "output"
    if isinstance(output_field, dict):
        return output_field
    if isinstance(output_field, str):
        try:
            return json.loads(output_field)
        except json.JSONDecodeError:
            return {}
    return {}

def normalize_emotions(parsed):
    # Target schema:
    # "emotions": [{"emotion": "<label>", "confidence_score": <0-1>}]
    #
    # Supports legacy input shapes:
    # 1) "emotion": "Neutral", "confidence_score": 1.0
    # 2) "emotion": ["Joy", "Surprise"], "confidence_score": 0.8
    # 3) "emotions": [{"emotion":"Joy","confidence_score":0.9}, ...]
    # 4) "emotions": ["Joy", "Surprise"]
    default_conf = float(parsed.get("confidence_score", 1.0))

    if isinstance(parsed.get("emotions"), list):
        out = []
        for item in parsed["emotions"]:
            if isinstance(item, dict):
                label = str(item.get("emotion", "")).strip()
                if not label:
                    continue
                conf = float(item.get("confidence_score", default_conf))
                out.append({"emotion": label, "confidence_score": conf})
            else:
                label = str(item).strip()
                if label:
                    out.append({"emotion": label, "confidence_score": default_conf})
        if out:
            return out

    legacy = parsed.get("emotion", "Neutral")
    labels = []
    if legacy is None:
        labels = ["Neutral"]
    elif isinstance(legacy, list):
        labels = [str(e).strip() for e in legacy if str(e).strip()]
    else:
        e = str(legacy).strip()
        if "," in e:
            labels = [x.strip() for x in e.split(",") if x.strip()]
        elif e:
            labels = [e]

    if not labels:
        labels = ["Neutral"]
    return [{"emotion": label, "confidence_score": default_conf} for label in labels]

def convert_row(row):
    parsed = safe_parse_output(row.get("output"))

    polarity = parsed.get("polarity", "Neutral")
    emotions = normalize_emotions(parsed)
    reasoning = parsed.get("reasoning", "")

    # target row for full-model evaluation
    return {
        "instruction": row.get("instruction", "Analyze sentiment and emotions."),
        "input": row.get("input", ""),
        "expected_output": {
            "polarity": polarity,
            "emotions": emotions,
            "reasoning": reasoning
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="Old test jsonl path")
    parser.add_argument("--outfile", required=True, help="Converted test jsonl path")
    args = parser.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            new_row = convert_row(row)
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Converted {total} rows -> {out_path}")

if __name__ == "__main__":
    main()
