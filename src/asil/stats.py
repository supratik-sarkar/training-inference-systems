"""Latency statistics.

Percentiles use linear interpolation between order statistics. With the small
sample counts a laptop benchmark produces, p99 from a handful of requests is
not a meaningful number, and ``percentile`` records the sample count alongside
the value so a reader can judge it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile. None for an empty sample."""
    if not values:
        return None
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(slots=True)
class LatencySummary:
    """A sample summary that always carries its own sample size."""

    count: int
    mean_s: float | None
    p50_s: float | None
    p90_s: float | None
    p99_s: float | None
    min_s: float | None
    max_s: float | None

    @property
    def is_statistically_thin(self) -> bool:
        """True when the sample is too small for tail percentiles to mean much."""
        return self.count < 30

    def to_dict(self) -> dict[str, Any]:
        def r(v: float | None) -> float | None:
            return round(v, 6) if v is not None else None
        d = {"count": self.count, "mean_s": r(self.mean_s),
             "p50_s": r(self.p50_s), "p90_s": r(self.p90_s),
             "p99_s": r(self.p99_s), "min_s": r(self.min_s),
             "max_s": r(self.max_s)}
        if self.is_statistically_thin:
            d["caveat"] = (f"{self.count} samples; tail percentiles from this "
                           "few observations are indicative, not reliable")
        return d


def summarise(values: list[float]) -> LatencySummary:
    return LatencySummary(
        count=len(values), mean_s=mean(values),
        p50_s=percentile(values, 50), p90_s=percentile(values, 90),
        p99_s=percentile(values, 99),
        min_s=min(values) if values else None,
        max_s=max(values) if values else None,
    )


def tokens_per_second(completion_tokens: int, wall_time_s: float) -> float | None:
    """None rather than infinity when no time elapsed."""
    if wall_time_s <= 0 or completion_tokens <= 0:
        return None
    return completion_tokens / wall_time_s


__all__ = ["LatencySummary", "mean", "percentile", "summarise",
           "tokens_per_second"]
