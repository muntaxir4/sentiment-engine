# Instructions

- python3 -m venv .venv
- source .venv/bin/activate
- cd fine_tuning/

# Get Datasets

- pip install -r prepare_requirements.txt
- python prepare_balanced_dataset.py

# Train Model (Ollama Compatible)

- `pip install -r train_requirements.txt`

### Train for LoRA Adapters

- `pyhton train.py`

### Get a Merged Fine-Tuned model

- `python merge.py`

### Convert to GGUF using llama.cpp for Ollama

- `docker run --rm -v $(pwd):/data ghcr.io/ggerganov/llama.cpp:full-cuda \
--convert /data/qwen_merged_16bit \
--outtype q8_0 \
--outfile /data/sentiment-engine-q8.gguf
`

# Use with Ollama

- `ollama create sentiment-engine -f ./Modelfile`
- `ollama run sentiment-engine`
