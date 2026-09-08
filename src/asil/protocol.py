"""Backend protocol and request/response objects."""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from .capabilities import CapabilitySet


@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    priority: int = 0
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenEvent:
    """One emitted token with the monotonic time it was produced."""

    index: int
    text: str
    emitted_at: float


@dataclass(slots=True)
class GenerationResponse:
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    events: list[TokenEvent] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wall_time_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def time_to_first_token_s(self) -> float | None:
        """None when no token was emitted -- never zero as a stand-in."""
        if not self.events:
            return None
        return max(0.0, self.events[0].emitted_at - self.started_at)

    def inter_token_latencies_s(self) -> list[float]:
        return [
            max(0.0, b.emitted_at - a.emitted_at)
            for a, b in zip(self.events, self.events[1:], strict=False)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "backend": self.backend,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_time_s": round(self.wall_time_s, 6),
            "ttft_s": (round(self.time_to_first_token_s, 6)
                       if self.time_to_first_token_s is not None else None),
            "metadata": self.metadata,
        }


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> CapabilitySet: ...

    @property
    def is_real_inference(self) -> bool:
        """False for deterministic stand-ins.

        Present so no result file can be mistaken for a measurement of an
        actual model.
        """
        ...

    def available(self) -> bool: ...

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]: ...

    def memory_footprint(self) -> dict[str, Any]: ...


def now() -> float:
    return time.perf_counter()


__all__ = ["GenerationRequest", "GenerationResponse", "InferenceBackend",
           "TokenEvent", "now"]
