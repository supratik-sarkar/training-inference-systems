"""Dataset, registry, lifecycle, routing and the LoRA/QLoRA distinction."""
from __future__ import annotations

import pytest

from asal import (
    AdapterNotFound,
    AdapterNotServable,
    AdapterRecord,
    AdapterRegistry,
    AdapterRouter,
    AdapterStatus,
    BackendUnavailable,
    Dataset,
    Example,
    FakeTrainingBackend,
    MLXTrainingBackend,
    TrainingConfig,
    TrainingMethod,
    capability_report,
    seed_dataset,
)


def record(aid: str = "a1", ds: Dataset | None = None,
           method: TrainingMethod = TrainingMethod.LORA) -> AdapterRecord:
    ds = ds or seed_dataset()
    return AdapterRecord(aid, "base-model", ds.version, ds.content_hash,
                         TrainingConfig(method=method))


# -- dataset ---------------------------------------------------------------
def test_content_hash_is_order_independent() -> None:
    ds = seed_dataset()
    reversed_ds = Dataset("v1", list(reversed(ds.examples)))
    assert ds.content_hash == reversed_ds.content_hash


def test_content_hash_changes_with_content() -> None:
    ds = seed_dataset()
    mutated = Dataset("v1", [*ds.examples[:-1],
                             Example("x", "different", "output")])
    assert ds.content_hash != mutated.content_hash


def test_split_is_deterministic() -> None:
    a = seed_dataset().split()
    b = seed_dataset().split()
    assert [e.example_id for e in a[0].examples] == [e.example_id for e in b[0].examples]


def test_split_is_disjoint_and_complete() -> None:
    ds = seed_dataset()
    train, held = ds.split()
    assert not ({e.example_id for e in train.examples}
                & {e.example_id for e in held.examples})
    assert len(train) + len(held) == len(ds)


def test_split_membership_survives_dataset_growth() -> None:
    """A shuffle would move examples and invalidate every past comparison."""
    ds = seed_dataset()
    before = {e.example_id for e in ds.split()[1].examples}
    grown = Dataset("v2", [*ds.examples,
                           Example("new-1", "new prompt", "new completion")])
    after = {e.example_id for e in grown.split()[1].examples}
    assert before <= after or (before - after) == set()


def test_dataset_round_trips(tmp_path) -> None:
    ds = seed_dataset()
    loaded = Dataset.load(ds.write(tmp_path / "d.json"))
    assert loaded.content_hash == ds.content_hash
    assert len(loaded) == len(ds)


def test_jsonl_export_has_one_object_per_example() -> None:
    ds = seed_dataset()
    lines = ds.to_jsonl().splitlines()
    assert len(lines) == len(ds)
    import json
    assert all("text" in json.loads(line) for line in lines)


# -- registry and lifecycle ------------------------------------------------
def test_registration_and_lookup() -> None:
    reg = AdapterRegistry()
    reg.register(record("a1"))
    assert reg.get("a1").adapter_id == "a1"
    with pytest.raises(AdapterNotFound):
        reg.get("missing")


def test_duplicate_registration_is_rejected() -> None:
    reg = AdapterRegistry()
    reg.register(record("a1"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(record("a1"))


def test_unevaluated_adapter_cannot_be_activated() -> None:
    """An unevaluated adapter is an unknown quantity, not a fast path."""
    reg = AdapterRegistry()
    reg.register(record("a1"))
    with pytest.raises(AdapterNotServable, match="only an evaluated"):
        reg.activate("a1")


def test_activation_after_evaluation_succeeds() -> None:
    reg = AdapterRegistry()
    r = reg.register(record("a1"))
    r.transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    assert reg.active_id == "a1"
    assert reg.active().status is AdapterStatus.ACTIVE


def test_hotswap_deactivates_the_previous_adapter() -> None:
    reg = AdapterRegistry()
    for aid in ("a1", "a2"):
        r = reg.register(record(aid))
        r.transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    reg.activate("a2")
    assert reg.active_id == "a2"
    assert reg.get("a1").status is AdapterStatus.EVALUATED


def test_retiring_the_active_adapter_clears_it() -> None:
    reg = AdapterRegistry()
    r = reg.register(record("a1"))
    r.transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    reg.retire("a1")
    assert reg.active_id is None
    assert reg.get("a1").status is AdapterStatus.RETIRED


def test_history_records_every_transition() -> None:
    r = record("a1")
    r.transition(AdapterStatus.TRAINING)
    r.transition(AdapterStatus.TRAINED)
    assert [h["to"] for h in r.history] == ["training", "trained"]


def test_lookup_by_dataset_hash() -> None:
    ds = seed_dataset()
    reg = AdapterRegistry()
    reg.register(record("a1", ds))
    reg.register(record("a2", ds))
    assert len(reg.by_dataset(ds.content_hash)) == 2


# -- routing ---------------------------------------------------------------
def test_router_falls_back_to_the_active_adapter() -> None:
    reg = AdapterRegistry()
    r = reg.register(record("a1"))
    r.transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    assert AdapterRouter(reg).resolve(["anything"]).adapter_id == "a1"


def test_router_matches_a_tag_rule() -> None:
    reg = AdapterRegistry()
    for aid in ("a1", "a2"):
        reg.register(record(aid)).transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    router = AdapterRouter(reg)
    router.add_rule("json", "a2")
    assert router.resolve(["json"]).adapter_id == "a2"


def test_router_skips_a_rule_pointing_at_an_unservable_adapter() -> None:
    reg = AdapterRegistry()
    reg.register(record("a1")).transition(AdapterStatus.EVALUATED)
    reg.register(record("a2"))
    reg.activate("a1")
    router = AdapterRouter(reg)
    router.add_rule("json", "a2")
    assert router.resolve(["json"]).adapter_id == "a1"


def test_router_explains_its_choice() -> None:
    reg = AdapterRegistry()
    reg.register(record("a1")).transition(AdapterStatus.EVALUATED)
    reg.activate("a1")
    e = AdapterRouter(reg).explain(["x"])
    assert e["via"] == "active_default" and e["adapter_id"] == "a1"


# -- training backends -----------------------------------------------------
def test_fake_backend_declares_it_trains_nothing() -> None:
    b = FakeTrainingBackend()
    assert b.is_real_training is False
    ok, reason = b.supports(TrainingMethod.LORA)
    assert ok and "trains nothing" in reason


def test_fake_training_marks_its_artifact_clearly(tmp_path) -> None:
    r = record("a1")
    result = FakeTrainingBackend().train(r, seed_dataset(), tmp_path)
    assert result.is_real_training is False
    assert result.final_loss is None, "a fake run must not report a loss"
    assert "NOT_A_REAL_ADAPTER" in result.artifact_path
    assert r.status is AdapterStatus.TRAINED


def test_mlx_backend_is_unavailable_here() -> None:
    b = MLXTrainingBackend()
    if b.available():
        pytest.skip("MLX present")
    assert b.is_real_training is True
    with pytest.raises(BackendUnavailable, match="Apple Silicon"):
        b.train(record("a1"), seed_dataset(), __import__("pathlib").Path("/tmp"))


def test_lora_and_qlora_are_reported_separately() -> None:
    """QLoRA must not inherit LoRA's status."""
    b = MLXTrainingBackend()
    if b.available():
        pytest.skip("MLX present")
    lora_ok, lora_reason = b.supports(TrainingMethod.LORA)
    qlora_ok, qlora_reason = b.supports(TrainingMethod.QLORA)
    assert lora_ok is False and qlora_ok is False
    assert "not importable" in lora_reason


def test_qlora_is_unverified_when_the_runtime_is_present(monkeypatch) -> None:
    """Even with MLX installed, QLoRA stays unverified until a run proves it."""
    b = MLXTrainingBackend()
    monkeypatch.setattr(MLXTrainingBackend, "available", lambda self: True)
    assert b.supports(TrainingMethod.LORA)[0] is True
    ok, reason = b.supports(TrainingMethod.QLORA)
    assert ok is False
    assert "UNVERIFIED" in reason


def test_qlora_becomes_supported_only_after_a_verified_run(monkeypatch) -> None:
    """The positive branch: the flag is settable, but only by a real run."""
    b = MLXTrainingBackend()
    monkeypatch.setattr(MLXTrainingBackend, "available", lambda self: True)
    assert b.supports(TrainingMethod.QLORA)[0] is False
    b._qlora_verified = True
    ok, reason = b.supports(TrainingMethod.QLORA)
    assert ok is True and "verified by a completed run" in reason


def test_qlora_config_sets_quantization_bits() -> None:
    cfg = TrainingConfig(method=TrainingMethod.QLORA)
    assert cfg.quantization_bits == 4
    assert TrainingConfig(method=TrainingMethod.LORA).quantization_bits is None


def test_capability_report_covers_both_methods() -> None:
    report = capability_report()
    for info in report.values():
        assert set(info["methods"]) == {"lora", "qlora"}
        assert all(m["reason"] for m in info["methods"].values())
