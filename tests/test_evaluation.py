"""Evaluation, regression comparison, failure mining and cycle-2 datasets."""
from __future__ import annotations

import json

import pytest

from asal import (
    Dataset,
    Example,
    SyntheticEvaluationRefused,
    build_next_dataset,
    compare,
    evaluate,
    mine_failures,
    seed_dataset,
)


def perfect(prompt: str) -> str:
    ds = seed_dataset()
    return next(e.completion for e in ds.examples if e.prompt == prompt)


def broken_json(prompt: str) -> str:
    return "{not valid json"


def wrong_team(prompt: str) -> str:
    d = json.loads(perfect(prompt))
    d["team"] = "wrong"
    return json.dumps(d)


# -- evaluation ------------------------------------------------------------
def test_perfect_predictor_scores_one() -> None:
    report = evaluate("a1", seed_dataset(), perfect)
    assert report.exact_match_rate == 1.0
    assert report.json_validity_rate == 1.0
    assert report.failures() == []


def test_invalid_json_is_detected() -> None:
    report = evaluate("a1", seed_dataset(), broken_json)
    assert report.json_validity_rate == 0.0
    assert report.exact_match_rate == 0.0


def test_field_accuracy_isolates_the_wrong_field() -> None:
    acc = evaluate("a1", seed_dataset(), wrong_team).field_accuracy()
    assert acc["team"] == 0.0
    assert acc["ticket"] == 1.0
    assert acc["status"] == 1.0


def test_report_binds_to_the_dataset_hash() -> None:
    ds = seed_dataset()
    assert evaluate("a1", ds, perfect).dataset_hash == ds.content_hash


def test_fake_evaluation_is_not_a_real_measurement() -> None:
    report = evaluate("a1", seed_dataset(), perfect)
    assert report.is_real_measurement is False
    assert "NOT a model evaluation" in report.to_dict()["disclaimer"]


def test_writing_a_required_real_evaluation_is_refused(tmp_path) -> None:
    report = evaluate("a1", seed_dataset(), perfect)
    with pytest.raises(SyntheticEvaluationRefused, match="refusing to write"):
        report.write(tmp_path / "e.json", require_real=True)


def test_writing_without_the_flag_is_labelled(tmp_path) -> None:
    p = evaluate("a1", seed_dataset(), perfect).write(tmp_path / "e.json")
    assert json.loads(p.read_text())["is_real_measurement"] is False


# -- regression ------------------------------------------------------------
def test_identical_reports_pass_the_gate() -> None:
    ds = seed_dataset()
    a = evaluate("a1", ds, perfect)
    b = evaluate("a2", ds, perfect)
    assert compare(a, b).passed


def test_a_worse_candidate_fails() -> None:
    ds = seed_dataset()
    result = compare(evaluate("a1", ds, perfect), evaluate("a2", ds, wrong_team))
    assert not result.passed
    assert any("exact match fell" in r for r in result.reasons)


def test_field_regression_is_named() -> None:
    ds = seed_dataset()
    result = compare(evaluate("a1", ds, perfect), evaluate("a2", ds, wrong_team))
    assert "team" in result.detail["regressed_fields"]


def test_comparison_across_datasets_is_refused() -> None:
    """An improvement measured on different questions is not an improvement."""
    a = evaluate("a1", seed_dataset(), perfect)
    other = Dataset("v9", [Example("x", "different prompt", "different")])
    b = evaluate("a2", other, lambda p: "different")
    result = compare(a, b)
    assert not result.passed
    assert any("different datasets" in r for r in result.reasons)


def test_json_validity_regression_is_caught() -> None:
    ds = seed_dataset()
    result = compare(evaluate("a1", ds, perfect),
                     evaluate("a2", ds, broken_json))
    assert not result.passed
    assert any("JSON validity fell" in r for r in result.reasons)


# -- failure mining --------------------------------------------------------
def test_invalid_json_clusters_separately() -> None:
    clusters = mine_failures(evaluate("a1", seed_dataset(), broken_json))
    labels = {c.label for c in clusters}
    assert "invalid_json" in labels
    cluster = next(c for c in clusters if c.label == "invalid_json")
    assert "formatting failure" in cluster.description


def test_wrong_field_clusters_by_field_name() -> None:
    clusters = mine_failures(evaluate("a1", seed_dataset(), wrong_team))
    assert any(c.label == "wrong_field:team" for c in clusters)
    assert not any(c.label == "wrong_field:ticket" for c in clusters)


def test_no_failures_yields_no_clusters() -> None:
    assert mine_failures(evaluate("a1", seed_dataset(), perfect)) == []


# -- cycle 2 ---------------------------------------------------------------
def test_next_dataset_adds_reinforcement_for_failures() -> None:
    base = seed_dataset()
    report = evaluate("a1", base, wrong_team)
    v2 = build_next_dataset(base, report, version="v2")
    assert len(v2) > len(base)
    reinforcement = [e for e in v2.examples if "reinforcement" in e.tags]
    assert len(reinforcement) == len(report.failures())
    assert v2.parent_version == base.version


def test_reinforcement_examples_carry_correct_completions() -> None:
    """The model is shown the right answer, not a fabricated new fact."""
    base = seed_dataset()
    v2 = build_next_dataset(base, evaluate("a1", base, wrong_team), version="v2")
    for e in (x for x in v2.examples if "reinforcement" in x.tags):
        original = next(o for o in base.examples if o.prompt == e.prompt)
        assert e.completion == original.completion
        assert e.metadata["reinforces"] == original.example_id


def test_next_dataset_has_a_different_hash() -> None:
    base = seed_dataset()
    v2 = build_next_dataset(base, evaluate("a1", base, wrong_team), version="v2")
    assert v2.content_hash != base.content_hash


def test_perfect_adapter_produces_no_reinforcement() -> None:
    base = seed_dataset()
    v2 = build_next_dataset(base, evaluate("a1", base, perfect), version="v2")
    assert len(v2) == len(base)
