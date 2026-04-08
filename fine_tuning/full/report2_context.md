# Report 2 Context (Sentiment Engine)

Use this as source context for generating the second project report.

## Project Snapshot

- Project: Sentiment Analysis using LLM
- Goal: Sentiment and emotion analysis from user text, with explainable reasoning.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Fine-tuning method: LoRA adapters (Unsloth + TRL SFT pipeline)
- Output format target: Structured JSON for API/frontend consumption

## Report 1 Baseline (Previous System)

- Dataset strategy: Balanced small dataset (~2300 training samples).
- Labeling style: Single-emotion output per sample.
- Reasoning: Not included in model output.
- Evaluation setup: Classification-style evaluation on balanced test split.
- Baseline metrics from stored eval:
  - Accuracy: `46.72%`
  - Macro F1: `0.45`
  - Weighted F1: `0.46`
  - Test support: `274`
- Baseline artifacts:
  - Eval text report: `fine_tuning/small/evals/eval_result.txt`
  - Confusion matrix: `fine_tuning/small/evals/evaluation_matrix.png`

## Report 2 Scope (Current System)

- Training moved from small balanced subset to full dataset pipeline.
- Task formulation upgraded from single-emotion classification to structured sentiment extraction.
- Model now predicts:
  - `polarity` (`Positive` / `Negative` / `Neutral` / `Mixed` when multi-emotion polarity conflicts)
  - `emotions` as a list of emotion-confidence pairs
  - `reasoning` text explaining prediction
- Data generation/training direction:
  - Full training data prepared through `fine_tuning/full/prepare_data_vllm.py`
  - Full training Colab flow in `fine_tuning/full/colab_train_full.ipynb`
  - Checkpointing and resume support enabled in training flow

## Evaluation Included In Report 2

Report 2 should explicitly cover the following evaluation dimensions for the upgraded multi-emotion + reasoning setup:

- Structured-output validity:
  - JSON parse success rate
  - Schema compliance rate (`polarity`, `emotions[]`, `reasoning` present)
- Emotion quality:
  - Exact match and/or overlap metrics for multi-emotion outputs
  - Precision/recall/F1 at emotion-label level
- Polarity quality:
  - Polarity accuracy (including handling of `Mixed` cases)
- Confidence behavior:
  - Confidence normalization checks (`sum(confidence_score)` per sample)
- Reasoning quality:
  - Consistency between predicted emotions and generated reasoning
  - Human spot-check examples (correct reasoning vs weak reasoning)
- Error analysis:
  - Common failure patterns by emotion group
  - Cases with ambiguous language / sarcasm / mixed sentiment

## Key Improvements To Highlight

- Data scale improvement:
  - From limited balanced set (~2300) to full-data training pipeline.
- Prediction richness:
  - From one label to multi-emotion probabilistic output.
- Explainability:
  - Added reasoning generation for each analysis.
- Robustness and operability:
  - Training checkpointing + resume for long runs.
  - Full-model merge and GGUF conversion path for deployment.
- Product integration:
  - Frontend updated to show multiple sentiment cards with per-item confidence and reasoning.

## Frontend/Product Updates To Mention

- UI now supports multiple detected sentiments in a single response.
- Each detected sentiment is shown with:
  - polarity
  - emotion label
  - confidence bar/percentage
  - reasoning text
- Dedicated reasoning display component is integrated in analysis flow.
- Result section clearly indicates the number of detected sentiments.

## Suggested Report 2 Structure

1. Problem and limitations of Report 1 system.
2. System redesign for full dataset + multi-emotion + reasoning.
3. Training and infrastructure updates (Colab, checkpoints, merge, GGUF).
4. Evaluation methodology and metrics for structured outputs.
5. Comparative discussion: Report 1 baseline vs Report 2 system.
6. Product impact (frontend and UX improvements).
7. Limitations and next steps.

## Verifiable Repo References (Extras, could be ignored as these are not verified)

- `fine_tuning/train.py` (full dataset config + checkpoints/resume)
- `fine_tuning/full/prepare_data_vllm.py` (full data preparation with reasoning JSON)
- `fine_tuning/full/colab_train_full.ipynb` (Drive-based full training flow)
- `fine_tuning/merge.py` (merge adapters into full model)
- `fine_tuning/full/convert_to_gguf_docker.sh` (GGUF conversion via Docker + llama.cpp)
- `fine_tuning/small/evals/eval_result.txt` (Report 1 baseline metrics)
- `frontend/components/sentiment-analyzer.tsx` (multi-result UX flow)
- `frontend/components/sentiment-card.tsx` (polarity/emotion/confidence/reasoning card UI)
- `frontend/components/reasoning-display.tsx` (reasoning view component)
