"""Backend capability declaration.

The central rule of this repository: a backend advertises what it actually
supports, and the benchmark suite skips everything else with a stated reason.

Nothing is simulated to make a feature list look longer. There is no fake
paged KV cache, no fake continuous batching, no fake speculative decoding. If
a capability is absent, the corresponding measurement does not exist -- it is
not estimated, interpolated or borrowed from a similar backend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Capability(StrEnum):
    STREAMING = "streaming"
    PREFIX_CACHE = "prefix_cache"
    SPECULATIVE_DECODING = "speculative_decoding"
    QUANTIZATION = "quantization"
    CONCURRENCY = "concurrency"
    MEMORY_METRICS = "memory_metrics"
    TOKEN_COUNTS = "token_counts"


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    """What a backend can actually do, declared explicitly."""

    streaming: bool = False
    prefix_cache: bool = False
    speculative_decoding: bool = False
    quantization: bool = False
    concurrency: bool = False
    memory_metrics: bool = False
    token_counts: bool = False

    def supports(self, cap: Capability) -> bool:
        return bool(getattr(self, str(cap)))

    def supported(self) -> list[Capability]:
        return [c for c in Capability if self.supports(c)]

    def unsupported(self) -> list[Capability]:
        return [c for c in Capability if not self.supports(c)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnsupportedCapability(RuntimeError):
    """Raised when a caller asks a backend for something it does not do."""

    def __init__(self, backend: str, cap: Capability, reason: str = "") -> None:
        self.backend, self.capability, self.reason = backend, cap, reason
        super().__init__(
            f"backend {backend!r} does not support {cap}"
            + (f": {reason}" if reason else "")
        )


class BackendUnavailable(RuntimeError):
    """Raised when a backend's runtime is not installed on this machine."""


class SyntheticResultRefused(RuntimeError):
    """Raised when something tries to record a measurement that was not measured.

    This exists because the tempting failure in a benchmarking repository is
    not lying outright -- it is emitting a plausible number when a run did not
    happen, so a chart has no gap in it. That path is closed in code.
    """


__all__ = ["BackendUnavailable", "Capability", "CapabilitySet",
           "SyntheticResultRefused", "UnsupportedCapability"]
