#!/usr/bin/env bash
set -euo pipefail

# Convert merged HF model -> GGUF using llama.cpp Docker image.
# Run from repository root or any directory.
#
# Example:
#   bash fine_tuning/convert_to_gguf_docker.sh q8_0
#   bash fine_tuning/convert_to_gguf_docker.sh f16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINE_TUNING_DIR="${SCRIPT_DIR}"
FULL_DIR="${FINE_TUNING_DIR}/full"

MERGED_MODEL_DIR="${FULL_DIR}/qwen_merged_full_16bit"
OUTTYPE="${1:-q8_0}"
OUTPUT_FILE="${FULL_DIR}/sentiment-engine-full-${OUTTYPE}.gguf"

if [[ ! -d "${MERGED_MODEL_DIR}" ]]; then
  echo "Merged model directory not found: ${MERGED_MODEL_DIR}"
  echo "Run: python3 ${FINE_TUNING_DIR}/merge.py"
  exit 1
fi

echo "Converting model to GGUF..."
echo "Input:  ${MERGED_MODEL_DIR}"
echo "Output: ${OUTPUT_FILE}"
echo "Type:   ${OUTTYPE}"

docker run --rm \
  -v "${FINE_TUNING_DIR}:/data" \
  ghcr.io/ggerganov/llama.cpp:full-cuda \
  --convert /data/full/qwen_merged_full_16bit \
  --outtype "${OUTTYPE}" \
  --outfile "/data/full/$(basename "${OUTPUT_FILE}")"

echo "Done: ${OUTPUT_FILE}"
