"""Training backends.

``FakeTrainingBackend`` exercises the orchestration -- lifecycle transitions,
registry updates, artifact bookkeeping -- without a model. It declares
``is_real_training = False`` and every record it produces carries that flag.

``MLXTrainingBackend`` is written against mlx-lm and has NEVER been executed:
MLX requires Apple Silicon. It probes the runtime and raises rather than
falling back, and its QLoRA support is reported as unverified until a real run
proves otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .adapters import AdapterRecord, AdapterStatus, TrainingMethod
from .dataset import Dataset


class BackendUnavailable(RuntimeError):
    """The backend's runtime is not installed on this machine."""


class SyntheticTrainingRefused(RuntimeError):
    """Raised when something tries to record training that did not happen."""


@dataclass(slots=True)
class TrainingResult:
    adapter_id: str
    backend: str
    is_real_training: bool
    iterations: int
    final_loss: float | None
    artifact_path: str | None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "backend": self.backend,
                "is_real_training": self.is_real_training,
                "iterations": self.iterations, "final_loss": self.final_loss,
                "artifact_path": self.artifact_path, "detail": self.detail}


@runtime_checkable
class TrainingBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_real_training(self) -> bool: ...

    def available(self) -> bool: ...

    def supports(self, method: TrainingMethod) -> tuple[bool, str]: ...

    def train(self, record: AdapterRecord, dataset: Dataset,
              output_dir: Path) -> TrainingResult: ...


@dataclass(slots=True)
class FakeTrainingBackend:
    """Orchestration-only backend. Produces no weights and claims none."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def is_real_training(self) -> bool:
        return False

    def available(self) -> bool:
        return True

    def supports(self, method: TrainingMethod) -> tuple[bool, str]:
        return True, ("orchestration only; this backend trains nothing, so it "
                      "'supports' every method in the bookkeeping sense and "
                      "none in the useful one")

    def train(self, record: AdapterRecord, dataset: Dataset,
              output_dir: Path) -> TrainingResult:
        record.transition(AdapterStatus.TRAINING, f"fake backend, {len(dataset)} examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / f"{record.adapter_id}.NOT_A_REAL_ADAPTER.json"
        marker.write_text(json.dumps({
            "adapter_id": record.adapter_id,
            "warning": "No weights were trained. This file exists so the "
                       "registry lifecycle can be tested end to end.",
            "dataset_hash": dataset.content_hash,
            "config": record.config.to_dict(),
        }, indent=2, sort_keys=True))
        record.artifact_path = str(marker)
        record.transition(AdapterStatus.TRAINED, "fake training complete")
        return TrainingResult(
            adapter_id=record.adapter_id, backend=self.name,
            is_real_training=False, iterations=record.config.iterations,
            final_loss=None,
            artifact_path=str(marker),
            detail={"note": "no weights produced; final_loss is null rather "
                            "than a plausible number"},
        )


@dataclass(slots=True)
class MLXTrainingBackend:
    """LoRA/QLoRA via `mlx_lm.lora`. NEVER EXECUTED in this repository."""

    base_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    _qlora_verified: bool = field(default=False, repr=False)

    @property
    def name(self) -> str:
        return f"mlx-lm:{self.base_model}"

    @property
    def is_real_training(self) -> bool:
        return True

    def available(self) -> bool:
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError:
            return False
        return True

    def supports(self, method: TrainingMethod) -> tuple[bool, str]:
        """LoRA and QLoRA are reported separately, as the brief requires.

        LoRA is the documented path in mlx-lm. QLoRA additionally requires a
        quantised base model and a version of mlx-lm that trains against it;
        whether that works is a property of the installed version and the
        model, not of this code. It is therefore reported as UNVERIFIED until
        a Mac run sets the flag.
        """
        if not self.available():
            return False, ("mlx-lm not importable; neither method can be "
                           "verified on this machine")
        if method is TrainingMethod.LORA:
            return True, "mlx-lm exposes a documented LoRA training path"
        if self._qlora_verified:
            return True, "QLoRA verified by a completed run on this machine"
        return False, (
            "QLoRA support is UNVERIFIED. It depends on the installed mlx-lm "
            "version and a quantised base model. This repository does not "
            "claim it until scripts/train_mac.sh completes a QLoRA run."
        )

    def _require(self, method: TrainingMethod) -> None:
        if not self.available():
            raise BackendUnavailable(
                "mlx / mlx-lm not importable. This backend requires Apple "
                "Silicon and has never been executed in this repository's CI. "
                "Run scripts/train_mac.sh on a Mac."
            )
        ok, reason = self.supports(method)
        if not ok:
            raise BackendUnavailable(f"{method} unavailable: {reason}")

    def train(self, record: AdapterRecord, dataset: Dataset,
              output_dir: Path) -> TrainingResult:
        self._require(record.config.method)
        cfg = record.config
        record.transition(AdapterStatus.TRAINING, f"mlx-lm {cfg.method}")
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            train, valid = dataset.split(eval_fraction=0.2)
            (data_dir / "train.jsonl").write_text(train.to_jsonl())
            (data_dir / "valid.jsonl").write_text(valid.to_jsonl())

            batch_size = min(cfg.batch_size, max(1, len(valid)))
            cmd = [
                sys.executable, "-m", "mlx_lm.lora", "--model", self.base_model,
                "--train", "--data", str(data_dir),
                "--iters", str(cfg.iterations),
                "--batch-size", str(batch_size),
                "--learning-rate", str(cfg.learning_rate),
                "--num-layers", str(cfg.layers_to_tune),
                "--adapter-path", str(output_dir / record.adapter_id),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True,  # noqa: S603
                                  timeout=7200, check=False)
        if proc.returncode != 0:
            record.transition(AdapterStatus.FAILED, proc.stderr[:200])
            raise RuntimeError(f"mlx-lm training failed: {proc.stderr[:400]}")

        path = output_dir / record.adapter_id
        record.artifact_path = str(path)
        record.transition(AdapterStatus.TRAINED, "mlx-lm training complete")
        return TrainingResult(
            adapter_id=record.adapter_id, backend=self.name,
            is_real_training=True, iterations=cfg.iterations,
            final_loss=_parse_final_loss(proc.stdout),
            artifact_path=str(path),
            detail={"method": str(cfg.method), "base_model": self.base_model},
        )


def _parse_final_loss(stdout: str) -> float | None:
    """Extract the last reported validation loss, or None if absent."""
    import re
    matches = re.findall(r"[Vv]al loss[:\s]+([0-9.]+)", stdout)
    return float(matches[-1]) if matches else None


def capability_report() -> dict[str, Any]:
    """What training is actually possible on this machine."""
    out: dict[str, Any] = {}
    for backend in (FakeTrainingBackend(), MLXTrainingBackend()):
        methods = {}
        for method in TrainingMethod:
            ok, reason = backend.supports(method)
            methods[str(method)] = {"supported": ok, "reason": reason}
        out[backend.name] = {
            "available": backend.available(),
            "is_real_training": backend.is_real_training,
            "methods": methods,
        }
    return out


__all__ = ["BackendUnavailable", "FakeTrainingBackend", "MLXTrainingBackend",
           "SyntheticTrainingRefused", "TrainingBackend", "TrainingResult",
           "capability_report"]
