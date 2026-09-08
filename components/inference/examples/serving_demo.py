"""Show the scheduler, capability gating and metric computation end to end.

    python examples/serving_demo.py

Runs against the deterministic backend, so nothing here is a performance
measurement, and the script says so.
"""
from __future__ import annotations

import asyncio

from asil.bench import run_benchmark

from asil import Capability, DeterministicBackend, GenerationRequest, Scheduler
from asil.serving import SchedulerConfig


async def main() -> int:
    backend = DeterministicBackend(ttft_s=0.01, itl_s=0.002)

    print("capabilities")
    caps = backend.capabilities()
    for cap in Capability:
        mark = "yes" if caps.has(cap) else "no "
        reason = "" if caps.has(cap) else f"  ({caps.why_not(cap)[:60]})"
        print(f"  {mark}  {cap}{reason}")

    print("\nscheduler")
    s = Scheduler(backend, SchedulerConfig(max_concurrency=4))
    print(f"  requested concurrency 4, effective {s.effective_concurrency}")
    await s.submit_all(
        [GenerationRequest(f"r{i}", f"prompt number {i}", max_tokens=16)
         for i in range(4)]
    )
    print(f"  completed {s.queue.stats.completed}, "
          f"failed {s.queue.stats.failed}, max depth {s.queue.stats.max_depth}")

    print("\nbenchmark matrix")
    report = await run_benchmark(backend)
    for cell in report.cells:
        name = cell["cell"]["name"]
        if cell["skipped"]:
            print(f"  SKIP  {name:<26} {cell['skip_reason'][:58]}")
        else:
            agg = cell["aggregate"]
            print(f"  ran   {name:<26} ttft_p50={agg['ttft_p50_s']:.4f}s "
                  f"tokens/s={agg['tokens_per_second_mean']:.1f}")

    print(f"\nsynthetic: {report.synthetic}")
    print("These timings come from configured constants, not a model. The")
    print("report writer refuses to persist them for exactly that reason.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
