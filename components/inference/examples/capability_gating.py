"""Show what each backend can do and what the suite therefore skips.

    python examples/capability_gating.py
"""
from __future__ import annotations

from asil import BACKENDS, DeterministicBackend, detect_available, run_benchmark


def main() -> int:
    print("backends on this machine\n")
    for key, info in detect_available().items():
        mark = "available" if info["available"] else "UNAVAILABLE"
        real = "real" if info["is_real_inference"] else "stand-in"
        print(f"  {key:<16} {mark:<12} {real}")
        print(f"      supports: {', '.join(info['supported']) or 'nothing'}")
    print()

    backend = DeterministicBackend(token_delay_s=0.0005,
                                   first_token_delay_s=0.002)
    report = run_benchmark(backend, requests=6, max_tokens=8)

    print(f"benchmark against {report.backend}\n")
    print(f"  {'case':<28} {'status':<10} reason")
    print("  " + "-" * 86)
    for c in report.cases:
        print(f"  {c['name']:<28} {c['status']:<10} {c['reason'][:50]}")

    print(f"\n  measured={report.measured}  skipped={report.skipped}")
    print(f"  is_performance_measurement: {report.is_performance_measurement}")
    print(f"\n  {report.to_dict()['disclaimer']}")

    print("\nno backend in this repository advertises these:")
    for key in BACKENDS:
        caps = BACKENDS[key]().capabilities
        assert not caps.speculative_decoding and not caps.prefix_cache
    print("  speculative_decoding, prefix_cache -- confirmed absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
