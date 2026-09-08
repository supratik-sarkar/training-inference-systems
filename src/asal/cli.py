"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import AdapterRecord, AdapterRegistry, TrainingConfig, TrainingMethod
from .dataset import Dataset, seed_dataset
from .evaluation import SyntheticEvaluationRefused, evaluate, mine_failures
from .training import FakeTrainingBackend, capability_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="asal", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("capabilities", help="what training is possible here")

    p = sub.add_parser("dataset", help="emit the seed dataset")
    p.add_argument("--version", default="v1")
    p.add_argument("--out", default=None)

    p = sub.add_parser("train", help="run the orchestration with a backend")
    p.add_argument("--adapter", required=True)
    p.add_argument("--method", choices=["lora", "qlora"], default="lora")
    p.add_argument("--backend", choices=["fake", "mlx"], default="fake")
    p.add_argument("--out", default="adapters")

    p = sub.add_parser("evaluate", help="evaluate a predictor over a dataset")
    p.add_argument("--adapter", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--require-real", action="store_true")

    args = ap.parse_args(argv)

    match args.cmd:
        case "capabilities":
            print(json.dumps(capability_report(), indent=2))
        case "dataset":
            ds = seed_dataset()
            ds.version = args.version
            print(f"version={ds.version} examples={len(ds)} "
                  f"hash={ds.content_hash[:16]}")
            train, held = ds.split()
            print(f"train={len(train)} eval={len(held)}")
            if args.out:
                print(f"wrote {ds.write(args.out)}")
        case "train":
            if args.backend == "mlx":
                from .training import MLXTrainingBackend
                backend = MLXTrainingBackend()
                ok, reason = backend.supports(TrainingMethod(args.method))
                if not ok:
                    print(f"cannot train: {reason}", file=sys.stderr)
                    return 2
            else:
                backend = FakeTrainingBackend()
            ds = seed_dataset()
            record = AdapterRecord(
                args.adapter, "base-model", ds.version, ds.content_hash,
                TrainingConfig(method=TrainingMethod(args.method)))
            AdapterRegistry().register(record)
            result = backend.train(record, ds, Path(args.out))
            print(json.dumps(result.to_dict(), indent=2))
        case "evaluate":
            ds = Dataset.load(args.dataset) if args.dataset else seed_dataset()
            lookup = {e.prompt: e.completion for e in ds.examples}
            report = evaluate(args.adapter, ds, lambda p: lookup.get(p, ""))
            print(json.dumps(report.metrics(), indent=2))
            print(f"real measurement: {report.is_real_measurement}")
            for c in mine_failures(report):
                print(f"  {c.label}: {len(c.example_ids)} examples")
            if args.out:
                try:
                    print(f"wrote {report.write(args.out, require_real=args.require_real)}")
                except SyntheticEvaluationRefused as exc:
                    print(f"REFUSED: {exc}", file=sys.stderr)
                    return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
