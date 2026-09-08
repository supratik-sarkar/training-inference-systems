"""Entry point executed under torchrun.

    torchrun --nproc_per_node=4 -m dtp.cli --out results/report.json
"""
from __future__ import annotations

import argparse
import sys

import torch.distributed as dist

from .experiments import EXPERIMENTS, run_experiment
from .harness import DistContext, process_group
from .report import Report, summarise


def _run(ctx: DistContext, names: list[str], out: str | None) -> int:
    results = [run_experiment(n, ctx) for n in names]
    dist.barrier()
    if not ctx.is_leader:
        return 0
    report = Report.build(ctx.world_size, tuple(ctx.mesh.mesh.shape), results)
    print(summarise(report))
    if out:
        path = report.write(out)
        print(f"\nwrote {path}")
    return 0 if report.all_passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dtp", description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run a subset of experiments by name")
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--list", action="store_true", help="list experiment names")
    args = ap.parse_args(argv)

    if args.list:
        for name in EXPERIMENTS:
            print(name)
        return 0

    names = args.only or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment(s): {', '.join(unknown)}")

    with process_group() as ctx:
        return _run(ctx, names, args.out)


if __name__ == "__main__":
    sys.exit(main())
