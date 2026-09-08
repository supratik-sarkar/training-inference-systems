#!/usr/bin/env bash
# Run the full experiment suite across N local CPU processes.
#   ./scripts/run_experiments.sh          # 4 ranks
#   ./scripts/run_experiments.sh 2        # 2 ranks
set -euo pipefail
NPROC="${1:-4}"
OUT="${2:-results/report-ws${NPROC}.json}"
echo "launching ${NPROC} gloo ranks on CPU"
torchrun --nproc_per_node="${NPROC}" --master_port=29540 \
         -m dtp.cli --out "${OUT}"
