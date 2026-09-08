"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys

from .backends import BACKENDS, detect_available
from .benchmark import run_benchmark
from .capabilities import SyntheticResultRefused
from .plotting import assert_plottable, load_results, plot_latency


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="asil", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("capabilities", help="probe backends available here")

    p = sub.add_parser("benchmark", help="run the capability-gated suite")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="deterministic")
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default=None)
    p.add_argument("--require-real", action="store_true",
                   help="refuse to write unless this is a real measurement")

    p = sub.add_parser("plot", help="plot a result file")
    p.add_argument("results")
    p.add_argument("--out", default="results/latency.png")

    p = sub.add_parser("verify", help="check whether a result file is a measurement")
    p.add_argument("results")

    args = ap.parse_args(argv)

    match args.cmd:
        case "capabilities":
            print(json.dumps(detect_available(), indent=2))
        case "benchmark":
            backend = BACKENDS[args.backend]()
            report = run_benchmark(backend, requests=args.requests,
                                   max_tokens=args.max_tokens,
                                   concurrency=args.concurrency)
            d = report.to_dict()
            print(f"backend: {report.backend}")
            print(f"real inference: {report.is_real_inference}")
            print(f"performance measurement: {report.is_performance_measurement}")
            print(f"\n{'case':<28} {'status':<10} reason")
            print("-" * 92)
            for c in report.cases:
                print(f"{c['name']:<28} {c['status']:<10} {c['reason'][:52]}")
            print(f"\nmeasured={report.measured} skipped={report.skipped}")
            if "disclaimer" in d:
                print(f"\n{d['disclaimer']}")
            if args.out:
                try:
                    print(f"\nwrote {report.write(args.out, require_real=args.require_real)}")
                except SyntheticResultRefused as exc:
                    print(f"\nREFUSED: {exc}", file=sys.stderr)
                    return 3
        case "plot":
            try:
                print(f"wrote {plot_latency(args.results, args.out)}")
            except SyntheticResultRefused as exc:
                print(f"REFUSED: {exc}", file=sys.stderr)
                return 3
        case "verify":
            report = load_results(args.results)
            try:
                assert_plottable(report)
                print("this file is a genuine performance measurement")
            except SyntheticResultRefused as exc:
                print(f"NOT a performance measurement: {exc}")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
