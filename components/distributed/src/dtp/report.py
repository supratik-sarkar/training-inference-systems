"""Machine-readable results.

Rank 0 writes one JSON document per run, carrying the exact environment so a
reader can tell what the numbers describe. Nothing is written unless the
experiments actually ran.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .experiments import ExperimentResult


@dataclass(slots=True)
class Report:
    world_size: int
    mesh_shape: list[int]
    backend: str
    results: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = 1

    @classmethod
    def build(cls, world_size: int, mesh_shape: tuple[int, ...],
              results: list[ExperimentResult]) -> Report:
        return cls(
            world_size=world_size,
            mesh_shape=list(mesh_shape),
            backend="gloo",
            results=[r.to_dict() for r in results],
            environment={
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or "unknown",
                "device": "cpu",
                "cuda_available": torch.cuda.is_available(),
                "mps_available": bool(
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ),
            },
        )

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r["passed"])

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results
                   if not r["passed"] and not r["skipped"])

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r["skipped"])

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p


def load_report(path: str | Path) -> Report:
    d = json.loads(Path(path).read_text())
    return Report(**d)


def summarise(report: Report) -> str:
    lines = [
        f"world_size={report.world_size}  mesh={report.mesh_shape}  "
        f"backend={report.backend}",
        f"torch={report.environment.get('torch')}  "
        f"device={report.environment.get('device')}",
        "",
    ]
    for r in report.results:
        if r["skipped"]:
            mark, note = "SKIP", r["skip_reason"]
        elif r["passed"]:
            mark, note = "PASS", ""
        else:
            mark, note = "FAIL", r.get("error") or ""
        lines.append(f"  {mark:<5} {r['name']:<34} {note}")
    lines += ["",
              f"passed={report.passed} failed={report.failed} "
              f"skipped={report.skipped}"]
    return "\n".join(lines)


__all__ = ["Report", "load_report", "summarise"]
