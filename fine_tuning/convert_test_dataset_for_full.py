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

def normalize_emotions(emotion_value):
    # convert old single label -> list for multi-emotion format
    if emotion_value is None:
        return []
    if isinstance(emotion_value, list):
        return [str(e).strip() for e in emotion_value if str(e).strip()]
    e = str(emotion_value).strip()
    if not e:
        return []
    # handle comma-separated legacy cases
    if "," in e:
        return [x.strip() for x in e.split(",") if x.strip()]
    return [e]

def convert_row(row):
    parsed = safe_parse_output(row.get("output"))

    polarity = parsed.get("polarity", "Neutral")
    emotions = normalize_emotions(parsed.get("emotion", "Neutral"))
    confidence = parsed.get("confidence_score", 1.0)
    reasoning = parsed.get("reasoning", "")

    # target row for full-model evaluation
    return {
        "instruction": row.get("instruction", "Analyze sentiment and emotions."),
        "input": row.get("input", ""),
        "expected_output": {
            "polarity": polarity,
            "emotions": emotions,              # <-- list now
            "confidence_score": confidence,
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
