"""Benchmark runner and result schema.

Two rules enforced in code rather than documentation:

1. A capability the backend does not declare is SKIPPED with a stated reason.
   It is never estimated and never silently omitted.
2. A result file from a backend where ``is_real_inference`` is False is marked
   as such and refuses to be labelled a performance measurement.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .capabilities import Capability, SyntheticResultRefused
from .protocol import GenerationRequest, InferenceBackend
from .serving import AdmissionPolicy, Scheduler
from .stats import summarise, tokens_per_second

SCHEMA_VERSION = 1


@dataclass(slots=True)
class CaseResult:
    name: str
    capability: str | None
    status: str                      # "measured" | "skipped" | "failed"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment() -> dict[str, Any]:
    """Recorded with every result so a number is never read out of context."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "is_apple_silicon": platform.machine() == "arm64"
        and platform.system() == "Darwin",
    }
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        env["mlx"] = getattr(mx, "__version__", "present")
    except ImportError:
        env["mlx"] = None
    return env


@dataclass(slots=True)
class BenchmarkReport:
    backend: str
    is_real_inference: bool
    capabilities: dict[str, bool]
    cases: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=environment)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = SCHEMA_VERSION

    @property
    def measured(self) -> int:
        return sum(1 for c in self.cases if c["status"] == "measured")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.cases if c["status"] == "skipped")

    @property
    def is_performance_measurement(self) -> bool:
        """Only true for a real backend on Apple Silicon."""
        return bool(self.is_real_inference
                    and self.environment.get("is_apple_silicon"))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["measured_cases"] = self.measured
        d["skipped_cases"] = self.skipped
        d["is_performance_measurement"] = self.is_performance_measurement
        if not self.is_performance_measurement:
            d["disclaimer"] = (
                "This file is NOT a performance measurement. It was produced "
                f"by backend {self.backend!r} with is_real_inference="
                f"{self.is_real_inference} on "
                f"{self.environment.get('machine')}. Timings describe the "
                "harness, not a model."
            )
        return d

    def write(self, path: str | Path, *,
              require_real: bool = False) -> Path:
        if require_real and not self.is_performance_measurement:
            raise SyntheticResultRefused(
                "refusing to write a performance result file: backend "
                f"{self.backend!r} is_real_inference={self.is_real_inference}, "
                f"machine={self.environment.get('machine')}. Run this on "
                "Apple Silicon with a real backend, or drop require_real."
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


DEFAULT_PROMPTS = [
    "Summarise the tradeoffs of append-only storage.",
    "Explain exponential backoff with full jitter.",
    "Describe what an idempotency key protects against.",
    "What does a hop count mean in graph traversal?",
]


def run_benchmark(backend: InferenceBackend, *, requests: int = 8,
                  max_tokens: int = 32,
                  concurrency: int = 4) -> BenchmarkReport:
    """Run every case the backend declares support for; skip the rest."""
    caps = backend.capabilities
    report = BenchmarkReport(
        backend=backend.name, is_real_inference=backend.is_real_inference,
        capabilities=caps.to_dict(),
    )

    def add(result: CaseResult) -> None:
        report.cases.append(result.to_dict())

    if not backend.available():
        add(CaseResult("availability", None, "skipped",
                       f"backend {backend.name!r} runtime is not present on "
                       "this machine; no case can run"))
        return report

    # -- serial latency ----------------------------------------------------
    sched = Scheduler(backend, AdmissionPolicy(), max_concurrency=1)
    for i in range(requests):
        sched.submit(GenerationRequest(
            f"serial-{i}", DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)],
            max_tokens=max_tokens))
    responses = sched.drain()
    if responses:
        add(CaseResult("serial_latency", None, "measured", "", {
            "requests": len(responses),
            "wall_time": summarise([r.wall_time_s for r in responses]).to_dict(),
            "tokens_per_second": summarise([
                tps for r in responses
                if (tps := tokens_per_second(r.completion_tokens,
                                             r.wall_time_s)) is not None
            ]).to_dict(),
        }))
    else:
        add(CaseResult("serial_latency", None, "failed",
                       "no responses returned"))

    # -- TTFT / ITL, streaming only ----------------------------------------
    if caps.supports(Capability.STREAMING):
        ttfts, itls = [], []
        for r in responses:
            if (t := r.time_to_first_token_s) is not None:
                ttfts.append(t)
            itls.extend(r.inter_token_latencies_s())
        add(CaseResult("time_to_first_token", str(Capability.STREAMING),
                       "measured" if ttfts else "failed", "",
                       {"ttft": summarise(ttfts).to_dict()}))
        add(CaseResult("inter_token_latency", str(Capability.STREAMING),
                       "measured" if itls else "failed", "",
                       {"itl": summarise(itls).to_dict()}))
    else:
        for case in ("time_to_first_token", "inter_token_latency"):
            add(CaseResult(case, str(Capability.STREAMING), "skipped",
                           f"{backend.name} does not support streaming; "
                           "per-token timings cannot be measured and are not "
                           "estimated"))

    # -- concurrency -------------------------------------------------------
    if caps.supports(Capability.CONCURRENCY):
        csched = Scheduler(backend, AdmissionPolicy(),
                           max_concurrency=concurrency)
        for i in range(requests):
            csched.submit(GenerationRequest(
                f"conc-{i}", DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)],
                max_tokens=max_tokens))
        cresponses = csched.drain()
        add(CaseResult("concurrent_throughput", str(Capability.CONCURRENCY),
                       "measured" if cresponses else "failed", "", {
                           "concurrency": concurrency,
                           "requests": len(cresponses),
                           "wall_time": summarise(
                               [r.wall_time_s for r in cresponses]).to_dict(),
                       }))
    else:
        add(CaseResult("concurrent_throughput", str(Capability.CONCURRENCY),
                       "skipped",
                       f"{backend.name} declares no concurrency support; "
                       "running it in parallel would measure contention, "
                       "not the backend"))

    # -- capabilities that are simply absent -------------------------------
    for cap, case in ((Capability.PREFIX_CACHE, "prefix_cache_reuse"),
                      (Capability.SPECULATIVE_DECODING, "speculative_speedup"),
                      (Capability.QUANTIZATION, "quantization_comparison")):
        if caps.supports(cap):
            add(CaseResult(case, str(cap), "skipped",
                           "backend declares support; this case requires a "
                           "Mac run with a real model and is not implemented "
                           "as a synthetic measurement"))
        else:
            add(CaseResult(case, str(cap), "skipped",
                           f"{backend.name} does not implement {cap}; no "
                           "measurement exists and none is estimated"))

    # -- memory ------------------------------------------------------------
    if caps.supports(Capability.MEMORY_METRICS):
        try:
            add(CaseResult("memory_footprint", str(Capability.MEMORY_METRICS),
                           "measured", "", backend.memory_footprint()))
        except Exception as exc:  # noqa: BLE001
            add(CaseResult("memory_footprint", str(Capability.MEMORY_METRICS),
                           "failed", f"{type(exc).__name__}: {exc}"))
    else:
        add(CaseResult("memory_footprint", str(Capability.MEMORY_METRICS),
                       "skipped",
                       f"{backend.name} exposes no memory counters"))

    return report


__all__ = ["BenchmarkReport", "CaseResult", "DEFAULT_PROMPTS",
           "SCHEMA_VERSION", "environment", "run_benchmark"]
