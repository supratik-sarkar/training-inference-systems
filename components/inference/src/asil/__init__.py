"""apple-silicon-inference-lab: local serving mechanics, honestly measured."""
from __future__ import annotations

from .backends import (
    BACKENDS,
    DeterministicBackend,
    LlamaCppBackend,
    MLXBackend,
    detect_available,
)
from .benchmark import (
    SCHEMA_VERSION,
    BenchmarkReport,
    CaseResult,
    environment,
    run_benchmark,
)
from .capabilities import (
    BackendUnavailable,
    Capability,
    CapabilitySet,
    SyntheticResultRefused,
    UnsupportedCapability,
)
from .plotting import assert_plottable, load_results, plot_latency
from .protocol import (
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
    TokenEvent,
)
from .serving import (
    AdmissionDecision,
    AdmissionPolicy,
    AdmissionResult,
    ModelRouter,
    PriorityQueue,
    Scheduler,
    SchedulerStats,
)
from .stats import LatencySummary, percentile, summarise, tokens_per_second

__version__ = "0.1.0"

__all__ = [
    "AdmissionDecision", "AdmissionPolicy", "AdmissionResult", "BACKENDS",
    "BackendUnavailable", "BenchmarkReport", "Capability", "CapabilitySet",
    "CaseResult", "DeterministicBackend", "GenerationRequest",
    "GenerationResponse", "InferenceBackend", "LatencySummary",
    "LlamaCppBackend", "MLXBackend", "ModelRouter", "PriorityQueue",
    "SCHEMA_VERSION", "Scheduler", "SchedulerStats", "SyntheticResultRefused",
    "TokenEvent", "UnsupportedCapability", "__version__", "assert_plottable",
    "detect_available", "environment", "load_results", "percentile",
    "plot_latency", "run_benchmark", "summarise", "tokens_per_second",
]
