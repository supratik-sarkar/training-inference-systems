"""The guards that stop a fake number becoming a chart.

    python examples/refuses_synthetic_results.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from asil import (
    DeterministicBackend,
    SyntheticResultRefused,
    assert_plottable,
    run_benchmark,
)


def main() -> int:
    report = run_benchmark(
        DeterministicBackend(token_delay_s=0.0, first_token_delay_s=0.0),
        requests=4, max_tokens=4,
    )
    print(f"backend: {report.backend}")
    print(f"is_real_inference: {report.is_real_inference}")
    print(f"is_performance_measurement: {report.is_performance_measurement}\n")

    tmp = Path(tempfile.mkdtemp()) / "r.json"

    print("1. writing with require_real=True")
    try:
        report.write(tmp, require_real=True)
        print("   ERROR: should have refused")
        return 1
    except SyntheticResultRefused as exc:
        print(f"   refused: {str(exc)[:96]}...")

    print("\n2. writing without the flag")
    path = report.write(tmp)
    print(f"   allowed, written to {path.name}, labelled:")
    import json
    print(f"   {json.loads(path.read_text())['disclaimer'][:96]}...")

    print("\n3. plotting it")
    try:
        assert_plottable(report.to_dict())
        print("   ERROR: should have refused")
        return 1
    except SyntheticResultRefused as exc:
        print(f"   refused: {str(exc)[:96]}...")

    print("\nAll three guards held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
