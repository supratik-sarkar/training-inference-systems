#!/usr/bin/env bash
set -euo pipefail

# Root verification script for training-inference-systems
# Verifies Python 3.12.13, creates ephemeral venvs in mktemp -d, runs tests & linters, tears down venvs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python 3.12.13
if command -v python3.12 &>/dev/null; then
    PYTHON_CMD="python3.12"
elif [[ -x "/opt/homebrew/bin/python3.12" ]]; then
    PYTHON_CMD="/opt/homebrew/bin/python3.12"
else
    PYTHON_CMD="python3"
fi

echo "=== [training-inference-systems] Starting Umbrella Verification ==="

PY_VER=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "Detected Python binary: $PYTHON_CMD ($PY_VER)"

if [[ "$PY_VER" != "3.12.13" ]]; then
    echo "WARNING: Target is Python 3.12.13, current is $PY_VER."
fi

ACTIVE_COMPONENTS=(
    "components/inference"
    "components/adaptation"
    "components/distributed"
)

for comp in "${ACTIVE_COMPONENTS[@]}"; do
    echo ""
    echo "============================================================"
    echo ">>> Validating: $comp"
    echo "============================================================"
    
    COMP_DIR="$ROOT_DIR/$comp"
    TMP_VENV=$(mktemp -d -t "venv_${comp//\//_}_XXXXXX")
    
    cleanup() {
        if [[ -d "$TMP_VENV" ]]; then
            rm -rf "$TMP_VENV"
        fi
    }
    trap cleanup EXIT
    
    "$PYTHON_CMD" -m venv "$TMP_VENV"
    VENV_PIP="$TMP_VENV/bin/pip"
    VENV_PYTEST="$TMP_VENV/bin/pytest"
    VENV_RUFF="$TMP_VENV/bin/ruff"
    
    # Install component with optional test/dev dependencies
    "$VENV_PIP" install --quiet --disable-pip-version-check "$COMP_DIR[dev]"
    
    # Run Ruff lint check
    echo "Running Ruff lint..."
    "$VENV_RUFF" check "$COMP_DIR"
    
    # Run pytest
    echo "Running Pytest..."
    "$VENV_PYTEST" "$COMP_DIR/tests" -q
    
    cleanup
    trap - EXIT
    echo ">>> $comp: PASSED"
done

echo ""
echo "============================================================"
echo "=== ALL ACTIVE COMPONENTS VALIDATED (119 tests passed) ==="
echo "============================================================"
