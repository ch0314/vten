#!/usr/bin/env python3
"""Run the hand-written Cocotb testbench on the passthrough DUT (Verilator).

Builds the DUT once (cached under ``build/cocotb_sim/``), then runs the
instrumented benchmark test (``cocotb_tb/bench_passthrough.py``) for a single
tensor size N and prints/writes a JSON result with phase timings.

Usage:
    python run_cocotb.py --n 4096 [--seed 42] [--json out.json] [--force-build]

Requires: cocotb==1.9.2 (see requirements.txt), Verilator >= 5.006 on PATH.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
DUT_SV = REPO_ROOT / "examples" / "passthrough" / "rtl" / "passthrough.sv"
BUILD_DIR = BENCH_DIR / "build" / "cocotb_sim"
TB_DIR = BENCH_DIR / "cocotb_tb"


def make_runner(force_build: bool = False):
    """Return a Verilator runner with build artifacts ready.

    cocotb 1.9's Verilator flow re-runs verilator+make unconditionally on
    every ``build()`` call, so when the cached binary already exists we skip
    the build and restore the runner state ``build()`` would have set
    (attribute names pinned to cocotb==1.9.2). Returns (runner, t_build_s).
    """
    from cocotb.runner import get_runner

    runner = get_runner("verilator")
    binary = BUILD_DIR / "passthrough"
    if binary.exists() and not force_build:
        runner.build_dir = BUILD_DIR
        runner.sources = [DUT_SV]
        runner.verilog_sources = []
        runner.vhdl_sources = []
        runner.hdl_toplevel = "passthrough"
        runner.waves = False
        return runner, 0.0

    t0 = perf_counter()
    runner.build(
        sources=[DUT_SV],
        hdl_toplevel="passthrough",
        build_dir=str(BUILD_DIR),
        waves=False,
    )
    return runner, perf_counter() - t0


def run(runner, n: int, seed: int = 42) -> dict:
    """Run the benchmark test for one tensor size. Returns the result dict."""
    if n % 32 != 0:
        raise SystemExit("N must be a multiple of 32 (whole 256-bit beats)")

    with tempfile.TemporaryDirectory(prefix="cocotb_bench_") as tmp:
        report_json = Path(tmp) / "report.json"
        t0 = perf_counter()
        results_xml = runner.test(
            hdl_toplevel="passthrough",
            test_module="bench_passthrough",
            test_dir=str(TB_DIR),
            build_dir=str(BUILD_DIR),
            results_xml=str(Path(tmp) / "results.xml"),
            extra_env={
                "BENCH_N": str(n),
                "BENCH_SEED": str(seed),
                "BENCH_RESULT_JSON": str(report_json),
                # Pin the embedded interpreter env to the one running this
                # script (a stale VIRTUAL_ENV otherwise leaks into the sim).
                "VIRTUAL_ENV": sys.prefix,
            },
        )
        t_total = perf_counter() - t0

        from cocotb.runner import get_results

        num_tests, num_failed = get_results(Path(results_xml))
        if num_failed or not report_json.exists():
            raise SystemExit(f"cocotb run failed ({num_failed}/{num_tests})")
        report = json.loads(report_json.read_text())

    report["t_total_s"] = t_total  # includes sim launch + cocotb startup
    report["framework"] = "cocotb"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1024, help="tensor elements (int8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, help="write result JSON here")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    os.chdir(BENCH_DIR)
    runner, t_build = make_runner(force_build=args.force_build)
    result = run(runner, args.n, args.seed)
    result["t_build_s"] = t_build

    out = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(out)
    print(out)


if __name__ == "__main__":
    main()
