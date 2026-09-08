#!/usr/bin/env bash
# Apple Silicon acceptance and benchmark run. Refuses to run anywhere else.
#
# This is the gate that moves 06-apple-silicon-inference-lab from
# MAC_VALIDATION_REQUIRED to a repository with real results.
set -euo pipefail

ARCH="$(uname -m)"
OS="$(uname -s)"
if [ "$ARCH" != "arm64" ] || [ "$OS" != "Darwin" ]; then
  echo "REFUSING TO RUN: ${OS}/${ARCH}, expected Darwin/arm64."
  echo
  echo "MLX requires Apple Silicon. This script will not produce a result"
  echo "file elsewhere, because a number measured on other hardware is not"
  echo "a measurement of this one."
  exit 2
fi

MODEL="${1:-mlx-community/Llama-3.2-1B-Instruct-4bit}"
echo "platform: ${OS}/${ARCH}"
echo "model:    ${MODEL}"
echo

python3 -c "import mlx.core, mlx_lm; print('mlx: available')" || {
  echo "MLX not importable. Install with: pip install -e '.[mac]'"
  exit 3
}

echo "--- capability probe ---"
asil capabilities

echo
echo "--- benchmark (real model, real hardware) ---"
SAFE_NAME="$(echo "${MODEL}" | tr '/:' '__')"
OUT="results/mlx-${SAFE_NAME}.json"

asil benchmark \
  --backend mlx \
  --requests 24 \
  --max-tokens 128 \
  --concurrency 1 \
  --out "${OUT}" \
  --require-real

echo
echo "--- verify the file is a genuine measurement ---"
asil verify "${OUT}"

echo
echo "--- plot ---"
pip install -q -e '.[plot]'
asil plot "${OUT}" --out "results/mlx-${SAFE_NAME}.png"

echo
echo "Wrote ${OUT}. Record the outcome in PORTFOLIO_VERIFICATION.yaml."
echo "Note which cases were SKIPPED: those capabilities remain unmeasured"
echo "and must not be described as supported."
