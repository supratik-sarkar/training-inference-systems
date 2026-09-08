"""Evaluation, regression comparison and failure mining.

Failure mining is the part that closes the loop: the examples an adapter got
wrong become the seed for the next dataset. That is the iteration cycle this
repository is actually about.
"""
from __future__ import annotations

import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import Dataset, Example

SCHEMA_VERSION = 1


class SyntheticEvaluationRefused(RuntimeError):
    """Raised when a result would be recorded without a real run behind it."""


@dataclass(slots=True)
class Prediction:
    example_id: str
    prompt: str
    expected: str
    actual: str
    tags: list[str] = field(default_factory=list)

    @property
    def exact_match(self) -> bool:
        return self.expected.strip() == self.actual.strip()

    def json_field_match(self) -> dict[str, bool]:
        """Per-field comparison when both sides parse as JSON objects."""
        try:
            exp = json.loads(self.expected)
            act = json.loads(self.actual)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(exp, dict) or not isinstance(act, dict):
            return {}
        return {k: act.get(k) == v for k, v in sorted(exp.items())}

    @property
    def is_valid_json(self) -> bool:
        try:
            json.loads(self.actual)
        except (json.JSONDecodeError, TypeError):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"example_id": self.example_id, "expected": self.expected,
                "actual": self.actual, "exact_match": self.exact_match,
                "is_valid_json": self.is_valid_json,
                "field_match": self.json_field_match(), "tags": self.tags}


@dataclass(slots=True)
class EvalReport:
    adapter_id: str
    dataset_version: str
    dataset_hash: str
    backend: str
    is_real_inference: bool
    predictions: list[Prediction] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    environment: dict[str, Any] = field(default_factory=lambda: {
        "python": platform.python_version(), "machine": platform.machine(),
        "platform": platform.platform(),
        "is_apple_silicon": platform.machine() == "arm64"
        and platform.system() == "Darwin",
    })
    schema_version: int = SCHEMA_VERSION

    @property
    def count(self) -> int:
        return len(self.predictions)

    @property
    def exact_match_rate(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.exact_match for p in self.predictions) / self.count

    @property
    def json_validity_rate(self) -> float:
        if not self.predictions:
            return 0.0
        return sum(p.is_valid_json for p in self.predictions) / self.count

    def field_accuracy(self) -> dict[str, float]:
        totals: dict[str, list[bool]] = {}
        for p in self.predictions:
            for field_name, ok in p.json_field_match().items():
                totals.setdefault(field_name, []).append(ok)
        return {k: round(sum(v) / len(v), 4) for k, v in sorted(totals.items())}

    def failures(self) -> list[Prediction]:
        return [p for p in self.predictions if not p.exact_match]

    @property
    def is_real_measurement(self) -> bool:
        return bool(self.is_real_inference
                    and self.environment.get("is_apple_silicon"))

    def metrics(self) -> dict[str, Any]:
        return {"count": self.count,
                "exact_match_rate": round(self.exact_match_rate, 4),
                "json_validity_rate": round(self.json_validity_rate, 4),
                "field_accuracy": self.field_accuracy(),
                "failures": len(self.failures())}

    def to_dict(self) -> dict[str, Any]:
        d = {"schema_version": self.schema_version,
             "adapter_id": self.adapter_id,
             "dataset_version": self.dataset_version,
             "dataset_hash": self.dataset_hash, "backend": self.backend,
             "is_real_inference": self.is_real_inference,
             "is_real_measurement": self.is_real_measurement,
             "created_at": self.created_at, "environment": self.environment,
             "metrics": self.metrics(),
             "predictions": [p.to_dict() for p in self.predictions]}
        if not self.is_real_measurement:
            d["disclaimer"] = (
                "NOT a model evaluation. Produced by backend "
                f"{self.backend!r} with is_real_inference="
                f"{self.is_real_inference} on {self.environment.get('machine')}. "
                "These numbers describe the harness, not an adapter."
            )
        return d

    def write(self, path: str | Path, *, require_real: bool = False) -> Path:
        if require_real and not self.is_real_measurement:
            raise SyntheticEvaluationRefused(
                f"refusing to write an evaluation result: backend "
                f"{self.backend!r}, is_real_inference={self.is_real_inference}, "
                f"machine={self.environment.get('machine')}"
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p


Predictor = Callable[[str], str]


def evaluate(adapter_id: str, dataset: Dataset, predict: Predictor, *,
             backend: str = "fake", is_real_inference: bool = False
             ) -> EvalReport:
    return EvalReport(
        adapter_id=adapter_id, dataset_version=dataset.version,
        dataset_hash=dataset.content_hash, backend=backend,
        is_real_inference=is_real_inference,
        predictions=[
            Prediction(e.example_id, e.prompt, e.completion, predict(e.prompt),
                       list(e.tags))
            for e in dataset.examples
        ],
    )


# -- regression comparison -------------------------------------------------
@dataclass(slots=True)
class RegressionResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(baseline: EvalReport, candidate: EvalReport, *,
            max_regression: float = 0.0,
            require_same_dataset: bool = True) -> RegressionResult:
    """Is the candidate adapter at least as good as the baseline?

    Comparing across different datasets is refused by default: an improvement
    measured on different questions is not an improvement.
    """
    reasons: list[str] = []
    if require_same_dataset and baseline.dataset_hash != candidate.dataset_hash:
        reasons.append(
            "evaluated against different datasets "
            f"({baseline.dataset_hash[:12]} vs {candidate.dataset_hash[:12]}); "
            "the comparison is not meaningful"
        )
        return RegressionResult(False, reasons,
                                {"baseline": baseline.metrics(),
                                 "candidate": candidate.metrics()})

    delta = candidate.exact_match_rate - baseline.exact_match_rate
    if delta < -max_regression:
        reasons.append(f"exact match fell by {abs(delta):.4f}")

    json_delta = candidate.json_validity_rate - baseline.json_validity_rate
    if json_delta < -max_regression:
        reasons.append(f"JSON validity fell by {abs(json_delta):.4f}")

    base_fields = baseline.field_accuracy()
    cand_fields = candidate.field_accuracy()
    regressed = {k: {"baseline": v, "candidate": cand_fields.get(k, 0.0)}
                 for k, v in base_fields.items()
                 if cand_fields.get(k, 0.0) < v - max_regression}
    if regressed:
        reasons.append(f"field accuracy regressed: {sorted(regressed)}")

    return RegressionResult(
        not reasons, reasons,
        {"exact_match_delta": round(delta, 4),
         "json_validity_delta": round(json_delta, 4),
         "regressed_fields": regressed,
         "baseline": baseline.metrics(), "candidate": candidate.metrics()},
    )


# -- failure mining --------------------------------------------------------
@dataclass(slots=True)
class FailureCluster:
    label: str
    example_ids: list[str]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mine_failures(report: EvalReport) -> list[FailureCluster]:
    """Group failures by their observable defect, not by guesswork."""
    invalid_json, wrong_fields, other = [], {}, []
    for p in report.failures():
        if not p.is_valid_json:
            invalid_json.append(p.example_id)
            continue
        fields = p.json_field_match()
        bad = [k for k, ok in fields.items() if not ok]
        if bad:
            for k in bad:
                wrong_fields.setdefault(k, []).append(p.example_id)
        else:
            other.append(p.example_id)

    clusters: list[FailureCluster] = []
    if invalid_json:
        clusters.append(FailureCluster(
            "invalid_json", sorted(invalid_json),
            "output did not parse as JSON; a formatting failure, not a "
            "knowledge failure"))
    for field_name, ids in sorted(wrong_fields.items()):
        clusters.append(FailureCluster(
            f"wrong_field:{field_name}", sorted(ids),
            f"JSON parsed but the {field_name!r} field was incorrect"))
    if other:
        clusters.append(FailureCluster(
            "text_mismatch", sorted(other),
            "parsed and fields matched, but the text differed"))
    return clusters


def build_next_dataset(base: Dataset, report: EvalReport, *,
                       version: str) -> Dataset:
    """Cycle-2 dataset: the originals plus targeted examples for each failure.

    The added examples are the *correct* completions for the prompts the
    adapter failed on. This is the honest version of the iteration loop -- no
    new knowledge is invented, the model is simply shown the right answer to
    the questions it got wrong.
    """
    failures = {p.example_id for p in report.failures()}
    reinforcement = [
        Example(f"{version}-reinforce-{i:03d}", e.prompt, e.completion,
                [*e.tags, "reinforcement"],
                {"reinforces": e.example_id,
                 "reason": "adapter produced an incorrect completion "
                           "for this prompt in the previous cycle"})
        for i, e in enumerate(
            e for e in base.examples if e.example_id in failures
        )
    ]
    return Dataset(
        version=version,
        examples=[*base.examples, *reinforcement],
        parent_version=base.version,
        notes=(f"cycle 2: {len(base.examples)} original examples plus "
               f"{len(reinforcement)} reinforcement examples mined from "
               f"failures of {report.adapter_id}"),
    )


__all__ = ["EvalReport", "FailureCluster", "Prediction", "Predictor",
           "RegressionResult", "SCHEMA_VERSION", "SyntheticEvaluationRefused",
           "build_next_dataset", "compare", "evaluate", "mine_failures"]
