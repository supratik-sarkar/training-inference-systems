"""Plotting that refuses to draw a chart from results that are not measurements.

A chart is the most persuasive artefact a benchmarking repository produces and
the easiest to produce dishonestly. The guard is here rather than in a review
checklist.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .capabilities import SyntheticResultRefused


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def assert_plottable(report: dict[str, Any]) -> None:
    """Raise unless this result file describes a genuine measurement."""
    if not report.get("is_performance_measurement"):
        raise SyntheticResultRefused(
            f"refusing to plot {report.get('backend')!r}: "
            f"is_real_inference={report.get('is_real_inference')}, "
            f"machine={report.get('environment', {}).get('machine')}. "
            "A chart drawn from harness timings would be read as model "
            "performance. Run scripts/benchmark_mac.sh on Apple Silicon."
        )
    measured = [c for c in report.get("cases", [])
                if c.get("status") == "measured"]
    if not measured:
        raise SyntheticResultRefused(
            "refusing to plot: no case in this file has status 'measured'"
        )


def plot_latency(path: str | Path, out: str | Path) -> Path:
    """Draw latency percentiles. Requires matplotlib and a real result file."""
    report = load_results(path)
    assert_plottable(report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed; install with pip install -e '.[plot]'"
        ) from exc

    cases = [c for c in report["cases"] if c["status"] == "measured"
             and "wall_time" in c.get("metrics", {})]
    if not cases:
        raise SyntheticResultRefused("no wall_time metrics to plot")

    labels = [c["name"] for c in cases]
    p50 = [c["metrics"]["wall_time"]["p50_s"] for c in cases]
    p90 = [c["metrics"]["wall_time"]["p90_s"] for c in cases]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], p50, width=0.4, label="p50")
    ax.bar([i + 0.2 for i in x], p90, width=0.4, label="p90")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title(f"{report['backend']} on {report['environment']['machine']}")
    ax.legend()
    fig.tight_layout()
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=120)
    plt.close(fig)
    return outp


__all__ = ["assert_plottable", "load_results", "plot_latency"]
