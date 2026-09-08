"""Adapter metadata, registry, lifecycle and routing.

An adapter is only useful if you can say what produced it. Every record binds
the adapter to its base model, its dataset content hash, its training method
and its hyperparameters -- so "which adapter is serving this request, and what
was it trained on" has an answer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class AdapterStatus(StrEnum):
    REGISTERED = "registered"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class TrainingMethod(StrEnum):
    """LoRA and QLoRA are tracked separately and deliberately.

    They are not interchangeable claims: QLoRA additionally requires a
    quantised base model and a runtime that supports training against it. This
    repository can express both, and asserts neither until a real run says so.
    """

    LORA = "lora"
    QLORA = "qlora"


@dataclass(slots=True)
class TrainingConfig:
    method: TrainingMethod = TrainingMethod.LORA
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    learning_rate: float = 1e-4
    batch_size: int = 4
    iterations: int = 100
    layers_to_tune: int = 8
    quantization_bits: int | None = None

    def __post_init__(self) -> None:
        if self.method is TrainingMethod.QLORA and self.quantization_bits is None:
            self.quantization_bits = 4

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["method"] = str(self.method)
        return d


@dataclass(slots=True)
class AdapterRecord:
    adapter_id: str
    base_model: str
    dataset_version: str
    dataset_hash: str
    config: TrainingConfig
    status: AdapterStatus = AdapterStatus.REGISTERED
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    history: list[dict[str, str]] = field(default_factory=list)

    def transition(self, status: AdapterStatus, note: str = "") -> None:
        self.history.append({
            "from": str(self.status), "to": str(status),
            "at": datetime.now(UTC).isoformat(), "note": note,
        })
        self.status = status

    @property
    def is_servable(self) -> bool:
        return self.status in (AdapterStatus.EVALUATED, AdapterStatus.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "base_model": self.base_model,
                "dataset_version": self.dataset_version,
                "dataset_hash": self.dataset_hash,
                "config": self.config.to_dict(), "status": str(self.status),
                "tags": self.tags, "metrics": self.metrics,
                "artifact_path": self.artifact_path,
                "created_at": self.created_at, "history": self.history}


class AdapterNotFound(KeyError):
    pass


class AdapterNotServable(RuntimeError):
    pass


@dataclass(slots=True)
class AdapterRegistry:
    """The set of known adapters, and which one is currently active."""

    adapters: dict[str, AdapterRecord] = field(default_factory=dict)
    active_id: str | None = None

    def register(self, record: AdapterRecord) -> AdapterRecord:
        if record.adapter_id in self.adapters:
            raise ValueError(f"adapter already registered: {record.adapter_id}")
        self.adapters[record.adapter_id] = record
        return record

    def get(self, adapter_id: str) -> AdapterRecord:
        try:
            return self.adapters[adapter_id]
        except KeyError as exc:
            raise AdapterNotFound(adapter_id) from exc

    def activate(self, adapter_id: str) -> AdapterRecord:
        """Hotswap. Only an evaluated adapter may be activated."""
        record = self.get(adapter_id)
        if not record.is_servable:
            raise AdapterNotServable(
                f"{adapter_id} has status {record.status}; only an evaluated "
                "adapter may serve traffic. An unevaluated adapter is an "
                "unknown quantity, not a fast path."
            )
        if self.active_id and self.active_id != adapter_id:
            previous = self.adapters[self.active_id]
            previous.transition(AdapterStatus.EVALUATED, "deactivated")
        record.transition(AdapterStatus.ACTIVE, "activated")
        self.active_id = adapter_id
        return record

    def retire(self, adapter_id: str) -> AdapterRecord:
        record = self.get(adapter_id)
        record.transition(AdapterStatus.RETIRED, "retired")
        if self.active_id == adapter_id:
            self.active_id = None
        return record

    def active(self) -> AdapterRecord | None:
        return self.adapters.get(self.active_id) if self.active_id else None

    def by_tag(self, tag: str) -> list[AdapterRecord]:
        return sorted((a for a in self.adapters.values() if tag in a.tags),
                      key=lambda a: a.adapter_id)

    def by_dataset(self, dataset_hash: str) -> list[AdapterRecord]:
        return sorted((a for a in self.adapters.values()
                       if a.dataset_hash == dataset_hash),
                      key=lambda a: a.adapter_id)

    def to_dict(self) -> dict[str, Any]:
        return {"active_id": self.active_id,
                "adapters": {k: v.to_dict()
                             for k, v in sorted(self.adapters.items())}}

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


@dataclass(slots=True)
class AdapterRouter:
    """Choose an adapter per request by tag, falling back to the active one."""

    registry: AdapterRegistry
    rules: list[tuple[str, str]] = field(default_factory=list)

    def add_rule(self, tag: str, adapter_id: str) -> None:
        self.registry.get(adapter_id)
        self.rules.append((tag, adapter_id))

    def resolve(self, tags: list[str]) -> AdapterRecord | None:
        for tag, adapter_id in self.rules:
            if tag in tags:
                record = self.registry.get(adapter_id)
                if record.is_servable:
                    return record
        return self.registry.active()

    def explain(self, tags: list[str]) -> dict[str, Any]:
        record = self.resolve(tags)
        matched = next((t for t, _ in self.rules if t in tags), None)
        return {"request_tags": tags, "matched_rule": matched,
                "adapter_id": record.adapter_id if record else None,
                "via": "rule" if matched else "active_default",
                "status": str(record.status) if record else None}


__all__ = ["AdapterNotFound", "AdapterNotServable", "AdapterRecord",
           "AdapterRegistry", "AdapterRouter", "AdapterStatus",
           "TrainingConfig", "TrainingMethod"]
