#!/usr/bin/env bash
# Apple Silicon LoRA/QLoRA training run. Refuses to run anywhere else.
#
# This is the gate that moves 07-apple-silicon-adapter-lab off
# MAC_VALIDATION_REQUIRED, and the only thing that may set qlora_verified.
set -euo pipefail

ARCH="$(uname -m)"; OS="$(uname -s)"
if [ "$ARCH" != "arm64" ] || [ "$OS" != "Darwin" ]; then
  echo "REFUSING TO RUN: ${OS}/${ARCH}, expected Darwin/arm64."
  echo "MLX requires Apple Silicon. No adapter weights will be produced here,"
  echo "and none will be simulated."
  exit 2
fi

METHOD="lora"
[ "${1:-}" = "--method" ] && METHOD="${2:-lora}"

echo "platform: ${OS}/${ARCH}   method: ${METHOD}"
python3 -c "import mlx.core, mlx_lm; print('mlx: available')" || {
  echo "MLX not importable. Install with: pip install -e '.[mac]'"; exit 3; }

echo
echo "--- capability probe ---"
asal capabilities

if [ "${METHOD}" = "qlora" ]; then
  echo
  echo "NOTE: QLoRA is reported UNVERIFIED until this run completes."
  echo "If mlx_lm.lora fails against the quantised base model, that is a"
  echo "real result: record qlora_support as UNSUPPORTED for this version"
  echo "and do NOT mark it supported in PORTFOLIO_VERIFICATION.yaml."
fi

echo
echo "--- dataset ---"
asal dataset --version v1 --out data/v1.json

echo
echo "--- training ---"
asal train --adapter "adapter-${METHOD}-v1" --method "${METHOD}" \
           --backend mlx --out adapters/

echo
echo "--- evaluation ---"
asal evaluate --adapter "adapter-${METHOD}-v1" --dataset data/v1.json \
              --out "results/eval-${METHOD}-v1.json" --require-real || true

echo
echo "Record the outcome in PORTFOLIO_VERIFICATION.yaml."
echo "If --require-real refused, the run was not a genuine measurement and"
echo "the status must stay MAC_VALIDATION_REQUIRED."
