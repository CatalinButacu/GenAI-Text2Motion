"""
Module Benchmark Suite - Python Runner
======================================
Alternative to batch file, runs all benchmarks and generates report.

Usage: py tests/benchmarks/run_all_benchmarks.py
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_benchmark(name: str, script: str) -> tuple:
    """Run a single benchmark and return (passed, time)."""
    print(f"\n{'=' * 70}")
    print(f"[{name}] Running...")
    print("=" * 70)

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        capture_output=False,
    )
    elapsed = time.time() - start

    passed = result.returncode == 0
    return passed, elapsed


def main():
    print("\n" + "=" * 70)
    print("           MODULE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(
        f"Directory: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}"
    )

    benchmarks = [
        ("M2 Scene Planner", "tests/benchmarks/benchmark_m2.py"),
        ("M4 Motion Generator", "tests/benchmarks/benchmark_m4.py"),
    ]

    results = []

    for name, script in benchmarks:
        passed, elapsed = run_benchmark(name, script)
        results.append((name, passed, elapsed))

    # Summary
    print("\n" + "=" * 70)
    print("                    BENCHMARK SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_time = 0

    for name, passed, elapsed in results:
        status = "[x] PASS" if passed else "[ ] FAIL"
        print(f"  {name:25} {status:10} ({elapsed:.1f}s)")
        if passed:
            total_passed += 1
        total_time += elapsed

    print("-" * 70)
    n = len(benchmarks)
    print(f"  {'TOTAL':25} {total_passed}/{n} passed  ({total_time:.1f}s)")
    print("=" * 70)

    if total_passed == n:
        print("\n ALL BENCHMARKS PASSED!")
        return 0
    else:
        print(f"\n {n - total_passed} BENCHMARK(S) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
