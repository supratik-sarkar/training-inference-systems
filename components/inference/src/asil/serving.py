"""Admission control, scheduling and concurrency.

This is the platform-independent half of a serving stack: what to accept,
what order to run it in, and how many at once. All of it is testable against
a deterministic backend, and none of it depends on the model.
"""
from __future__ import annotations

import heapq
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .capabilities import Capability, UnsupportedCapability
from .protocol import GenerationRequest, GenerationResponse, InferenceBackend, now


class AdmissionDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_QUEUE_FULL = "rejected_queue_full"
    REJECTED_TOO_LONG = "rejected_too_long"
    REJECTED_MALFORMED = "rejected_malformed"


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    decision: AdmissionDecision
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision is AdmissionDecision.ACCEPTED


@dataclass(slots=True)
class AdmissionPolicy:
    """Reject early rather than queueing work that will fail or time out.

    A queue that accepts everything converts overload into latency for
    everyone, instead of a clear rejection for a few.
    """

    max_queue_depth: int = 32
    max_prompt_tokens: int = 4096
    max_completion_tokens: int = 2048

    def admit(self, request: GenerationRequest,
              queue_depth: int) -> AdmissionResult:
        if not request.prompt.strip():
            return AdmissionResult(AdmissionDecision.REJECTED_MALFORMED,
                                   "empty prompt")
        if request.max_tokens <= 0:
            return AdmissionResult(AdmissionDecision.REJECTED_MALFORMED,
                                   f"max_tokens must be positive, got {request.max_tokens}")
        if request.max_tokens > self.max_completion_tokens:
            return AdmissionResult(
                AdmissionDecision.REJECTED_TOO_LONG,
                f"max_tokens {request.max_tokens} exceeds "
                f"{self.max_completion_tokens}")
        if len(request.prompt.split()) > self.max_prompt_tokens:
            return AdmissionResult(AdmissionDecision.REJECTED_TOO_LONG,
                                   "prompt exceeds max_prompt_tokens")
        if queue_depth >= self.max_queue_depth:
            return AdmissionResult(
                AdmissionDecision.REJECTED_QUEUE_FULL,
                f"queue depth {queue_depth} at limit {self.max_queue_depth}")
        return AdmissionResult(AdmissionDecision.ACCEPTED)


@dataclass(order=True, slots=True)
class _Entry:
    sort_key: tuple[int, int]
    request: GenerationRequest = field(compare=False)


@dataclass(slots=True)
class PriorityQueue:
    """Higher priority first; ties broken by arrival order, never by chance."""

    _heap: list[_Entry] = field(default_factory=list)
    _seq: int = 0

    def push(self, request: GenerationRequest) -> None:
        heapq.heappush(self._heap, _Entry((-request.priority, self._seq),
                                          request))
        self._seq += 1

    def pop(self) -> GenerationRequest:
        return heapq.heappop(self._heap).request

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def depth(self) -> int:
        return len(self._heap)


@dataclass(slots=True)
class SchedulerStats:
    admitted: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "rejected": self.rejected,
                "completed": self.completed, "failed": self.failed,
                "rejections": dict(sorted(self.rejections.items()))}


@dataclass(slots=True)
class Scheduler:
    """Runs queued requests against a backend, honouring its concurrency limit.

    A backend that does not declare ``concurrency`` is run strictly serially.
    Running a non-concurrent backend in parallel would produce timings that
    describe contention rather than the model.
    """

    backend: InferenceBackend
    policy: AdmissionPolicy = field(default_factory=AdmissionPolicy)
    max_concurrency: int = 4
    queue: PriorityQueue = field(default_factory=PriorityQueue)
    stats: SchedulerStats = field(default_factory=SchedulerStats)

    @property
    def effective_concurrency(self) -> int:
        if not self.backend.capabilities.supports(Capability.CONCURRENCY):
            return 1
        return max(1, self.max_concurrency)

    def submit(self, request: GenerationRequest) -> AdmissionResult:
        result = self.policy.admit(request, self.queue.depth)
        if result.accepted:
            self.queue.push(request)
            self.stats.admitted += 1
        else:
            self.stats.rejected += 1
            key = str(result.decision)
            self.stats.rejections[key] = self.stats.rejections.get(key, 0) + 1
        return result

    def drain(self) -> list[GenerationResponse]:
        """Run everything queued and return responses in completion order."""
        pending = [self.queue.pop() for _ in range(len(self.queue))]
        responses: list[GenerationResponse] = []
        lock = threading.Lock()
        errors: list[tuple[str, str]] = []

        def run(req: GenerationRequest) -> None:
            try:
                resp = self.backend.generate(req)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                with lock:
                    errors.append((req.request_id, f"{type(exc).__name__}: {exc}"))
                    self.stats.failed += 1
                return
            with lock:
                responses.append(resp)
                self.stats.completed += 1

        width = self.effective_concurrency
        for i in range(0, len(pending), width):
            batch = pending[i:i + width]
            if width == 1:
                for req in batch:
                    run(req)
            else:
                threads = [threading.Thread(target=run, args=(r,))
                           for r in batch]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

        self.stats.rejections.update(
            {f"error:{rid}": 1 for rid, _ in errors} if errors else {}
        )
        return responses

    def stream_one(self, request: GenerationRequest) -> Iterator[Any]:
        if not self.backend.capabilities.supports(Capability.STREAMING):
            raise UnsupportedCapability(
                self.backend.name, Capability.STREAMING,
                "this backend does not expose per-token output",
            )
        return self.backend.stream(request)


@dataclass(slots=True)
class ModelRouter:
    """Route requests to a backend by rule.

    Deliberately simple: rules are explicit predicates over the request, so a
    routing decision can be explained rather than guessed at.
    """

    default: InferenceBackend
    routes: list[tuple[str, InferenceBackend]] = field(default_factory=list)

    def add_route(self, tag: str, backend: InferenceBackend) -> None:
        self.routes.append((tag, backend))

    def resolve(self, request: GenerationRequest) -> tuple[str, InferenceBackend]:
        tag = request.metadata.get("route")
        for name, backend in self.routes:
            if name == tag:
                return name, backend
        return "default", self.default

    def explain(self, request: GenerationRequest) -> dict[str, Any]:
        name, backend = self.resolve(request)
        return {"request_id": request.request_id, "route": name,
                "backend": backend.name,
                "requested_route": request.metadata.get("route"),
                "is_real_inference": backend.is_real_inference}


__all__ = ["AdmissionDecision", "AdmissionPolicy", "AdmissionResult",
           "ModelRouter", "PriorityQueue", "Scheduler", "SchedulerStats", "now"]
