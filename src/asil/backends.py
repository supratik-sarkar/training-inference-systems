"""Backends: a deterministic stand-in, plus MLX and llama.cpp adapters.

Only the deterministic backend runs in CI. The two real ones are written and
never executed here -- MLX requires Apple Silicon, llama.cpp requires a built
binary and a GGUF model. Both probe their runtime and raise rather than
degrading to something that produces numbers.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .capabilities import (
    BackendUnavailable,
    Capability,
    CapabilitySet,
    UnsupportedCapability,
)
from .protocol import GenerationRequest, GenerationResponse, TokenEvent, now


def _count_tokens(text: str) -> int:
    """Whitespace token proxy. Not a real tokenizer, and not claimed to be."""
    return len(text.split())


@dataclass(slots=True)
class DeterministicBackend:
    """A fake generator with controllable timing, for testing everything else.

    It produces reproducible token streams so the scheduler, the queue, the
    streaming layer and the statistics can all be tested without a model.

    ``is_real_inference`` is False. Any result file produced with this backend
    records that, and the report writer refuses to mark such a run as a
    performance measurement.
    """

    token_delay_s: float = 0.001
    first_token_delay_s: float = 0.005
    fail_on: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def capabilities(self) -> CapabilitySet:
        # Honest about itself: it streams and counts tokens because it really
        # does. It declares no prefix cache, no speculative decoding and no
        # quantization, because it has none.
        return CapabilitySet(streaming=True, concurrency=True,
                             token_counts=True)

    @property
    def is_real_inference(self) -> bool:
        return False

    def available(self) -> bool:
        return True

    def _tokens(self, request: GenerationRequest) -> list[str]:
        seed = hashlib.blake2b(request.prompt.encode(), digest_size=8).digest()
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
        return [vocab[(seed[i % 8] + i) % len(vocab)]
                for i in range(request.max_tokens)]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.request_id in self.fail_on:
            raise RuntimeError(f"injected failure for {request.request_id}")
        started = now()
        events = list(self._emit(request, started))
        return GenerationResponse(
            request_id=request.request_id,
            text=" ".join(e.text for e in events),
            prompt_tokens=_count_tokens(request.prompt),
            completion_tokens=len(events), events=events,
            started_at=started, finished_at=now(), backend=self.name,
            metadata={"is_real_inference": False},
        )

    def _emit(self, request: GenerationRequest,
              started: float) -> Iterator[TokenEvent]:
        for i, tok in enumerate(self._tokens(request)):
            time.sleep(self.first_token_delay_s if i == 0 else self.token_delay_s)
            yield TokenEvent(i, tok, now())

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        if request.request_id in self.fail_on:
            raise RuntimeError(f"injected failure for {request.request_id}")
        yield from self._emit(request, now())

    def memory_footprint(self) -> dict[str, Any]:
        raise UnsupportedCapability(
            self.name, Capability.MEMORY_METRICS,
            "a deterministic stand-in has no model resident in memory; "
            "reporting a number here would be fabrication",
        )


@dataclass(slots=True)
class MLXBackend:
    """mlx-lm backend for Apple Silicon.

    NEVER EXECUTED in this repository. Written against the mlx_lm API. MLX
    requires Apple Silicon; on any other platform ``available()`` is False and
    every method raises ``BackendUnavailable``.

    Capabilities are declared from what mlx-lm genuinely provides. Speculative
    decoding is declared False rather than True-with-a-stub: mlx-lm's support
    varies by version, and asserting it without having run it would be exactly
    the kind of claim this repository exists to avoid.
    """

    model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"mlx:{self.model_id}"

    @property
    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            streaming=True, quantization=True, concurrency=False,
            memory_metrics=True, token_counts=True,
            prefix_cache=False, speculative_decoding=False,
        )

    @property
    def is_real_inference(self) -> bool:
        return True

    def available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError:
            return False
        return True

    def _require(self) -> None:
        if not self.available():
            raise BackendUnavailable(
                "mlx / mlx-lm not importable. This backend requires Apple "
                "Silicon and has never been executed in this repository's CI. "
                "Run scripts/benchmark_mac.sh on a Mac."
            )

    def _load(self) -> tuple[Any, Any]:
        self._require()
        if self._model is None:
            from mlx_lm import load  # type: ignore[import-not-found]
            self._model, self._tokenizer = load(self.model_id)
        return self._model, self._tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        events = []
        started = now()
        for event in self.stream(request):
            events.append(event)
        return GenerationResponse(
            request_id=request.request_id,
            text="".join(e.text for e in events),
            prompt_tokens=len(self._tokenizer.encode(request.prompt)),
            completion_tokens=len(events), events=events,
            started_at=started, finished_at=now(), backend=self.name,
            metadata={"is_real_inference": True, "model": self.model_id},
        )

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        model, tokenizer = self._load()
        from mlx_lm import stream_generate  # type: ignore[import-not-found]
        for i, chunk in enumerate(stream_generate(
            model, tokenizer, prompt=request.prompt,
            max_tokens=request.max_tokens,
        )):
            yield TokenEvent(i, getattr(chunk, "text", str(chunk)), now())

    def memory_footprint(self) -> dict[str, Any]:
        self._require()
        import mlx.core as mx  # type: ignore[import-not-found]
        return {
            "active_bytes": int(mx.get_active_memory()),
            "peak_bytes": int(mx.get_peak_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "source": "mlx.core memory counters",
        }


@dataclass(slots=True)
class LlamaCppBackend:
    """llama.cpp backend via the llama-cli binary.

    NEVER EXECUTED here. Requires a built llama.cpp and a GGUF model file.
    ``available()`` probes for the binary on PATH.

    Streaming is declared False: the subprocess interface used below returns
    output on completion, so per-token timestamps would be reconstructed after
    the fact rather than measured. Reconstructed timings are not measurements,
    and TTFT computed from them would be meaningless.
    """

    binary: str = "llama-cli"
    model_path: str = ""

    @property
    def name(self) -> str:
        return f"llama.cpp:{self.model_path or 'unset'}"

    @property
    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            streaming=False, quantization=True, concurrency=False,
            memory_metrics=False, token_counts=True,
            prefix_cache=False, speculative_decoding=False,
        )

    @property
    def is_real_inference(self) -> bool:
        return True

    def available(self) -> bool:
        return shutil.which(self.binary) is not None and bool(self.model_path)

    def _require(self) -> None:
        if not self.available():
            raise BackendUnavailable(
                f"{self.binary!r} not on PATH or model_path unset. This "
                "backend has never been executed in this repository's CI."
            )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self._require()
        started = now()
        proc = subprocess.run(  # noqa: S603
            [self.binary, "-m", self.model_path, "-p", request.prompt,
             "-n", str(request.max_tokens), "--temp", str(request.temperature)],
            capture_output=True, text=True, timeout=600, check=False,
        )
        finished = now()
        if proc.returncode != 0:
            raise RuntimeError(f"llama.cpp failed: {proc.stderr[:400]}")
        text = proc.stdout.strip()
        return GenerationResponse(
            request_id=request.request_id, text=text,
            prompt_tokens=_count_tokens(request.prompt),
            completion_tokens=_count_tokens(text), events=[],
            started_at=started, finished_at=finished, backend=self.name,
            metadata={"is_real_inference": True,
                      "note": "no per-token timestamps; TTFT is unavailable "
                              "from this interface and is reported as null"},
        )

    def stream(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        raise UnsupportedCapability(
            self.name, Capability.STREAMING,
            "the subprocess interface returns output on completion; "
            "per-token timings would be reconstructed, not measured",
        )

    def memory_footprint(self) -> dict[str, Any]:
        raise UnsupportedCapability(
            self.name, Capability.MEMORY_METRICS,
            "no in-process memory counters are exposed by the CLI",
        )


BACKENDS = {
    "deterministic": DeterministicBackend,
    "mlx": MLXBackend,
    "llamacpp": LlamaCppBackend,
}


def detect_available() -> dict[str, dict[str, Any]]:
    """Probe every backend's runtime on this machine."""
    out: dict[str, dict[str, Any]] = {}
    for key, cls in BACKENDS.items():
        backend = cls()
        out[key] = {
            "name": backend.name,
            "available": backend.available(),
            "is_real_inference": backend.is_real_inference,
            "capabilities": backend.capabilities.to_dict(),
            "supported": [str(c) for c in backend.capabilities.supported()],
        }
    return out


__all__ = ["BACKENDS", "DeterministicBackend", "LlamaCppBackend", "MLXBackend",
           "detect_available"]
