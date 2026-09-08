"""The full cycle: train, evaluate, mine failures, build cycle 2.

    python examples/iteration_cycle.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from asal import (
    AdapterRecord,
    AdapterRegistry,
    AdapterStatus,
    TrainingConfig,
    TrainingMethod,
    build_next_dataset,
    compare,
    evaluate,
    mine_failures,
    seed_dataset,
)
from asal.training import FakeTrainingBackend


def wrong_team(prompt: str) -> str:
    ds = seed_dataset()
    d = json.loads(next(e.completion for e in ds.examples if e.prompt == prompt))
    if d["team"] in ("storage", "security"):
        d["team"] = "platform"
    return json.dumps(d)


def perfect(prompt: str) -> str:
    ds = seed_dataset()
    return next(e.completion for e in ds.examples if e.prompt == prompt)


def main() -> int:
    ds = seed_dataset()
    print(f"dataset {ds.version}: {len(ds)} examples  "
          f"hash={ds.content_hash[:16]}")
    train, held = ds.split()
    print(f"  deterministic split: train={len(train)} eval={len(held)}\n")

    registry = AdapterRegistry()
    record = registry.register(AdapterRecord(
        "adapter-v1", "base-model", ds.version, ds.content_hash,
        TrainingConfig(method=TrainingMethod.LORA)))

    with tempfile.TemporaryDirectory() as tmp:
        result = FakeTrainingBackend().train(record, train, Path(tmp))
        print(f"training backend: {result.backend}")
        print(f"  is_real_training: {result.is_real_training}")
        print(f"  final_loss: {result.final_loss} (null, not invented)")
        print(f"  artifact: {Path(result.artifact_path).name}\n")

    baseline = evaluate("adapter-v1", ds, wrong_team)
    print(f"evaluation of {baseline.adapter_id}")
    for k, v in baseline.metrics().items():
        print(f"  {k}: {v}")

    print("\nfailure clusters")
    for c in mine_failures(baseline):
        print(f"  {c.label:<24} {len(c.example_ids)} examples")
        print(f"      {c.description}")

    v2 = build_next_dataset(ds, baseline, version="v2")
    print(f"\ncycle 2 dataset: {len(v2)} examples "
          f"(was {len(ds)}), hash={v2.content_hash[:16]}")
    print(f"  {v2.notes}")

    record.transition(AdapterStatus.EVALUATED, "cycle 1 evaluated")
    registry.activate("adapter-v1")
    print(f"\nactive adapter: {registry.active_id}")

    improved = evaluate("adapter-v2", ds, perfect)
    gate = compare(baseline, improved)
    print(f"\nregression gate v1 -> v2: {'PASS' if gate.passed else 'FAIL'}")
    print(f"  exact match delta: {gate.detail['exact_match_delta']:+.4f}")

    regressed = compare(improved, baseline)
    print(f"regression gate v2 -> v1: {'PASS' if regressed.passed else 'FAIL'}")
    for r in regressed.reasons:
        print(f"  {r}")
    return 0 if gate.passed and not regressed.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
