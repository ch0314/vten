#!/usr/bin/env python3
"""Run the passthrough tensor transfer through the vTen flow (Verilator).

Executes the same DUT/simulator/tensor-size combination as ``run_cocotb.py``
via vTen's SV-DPI shared-memory transport, and reports wall-clock phase
timings measured with ``time.perf_counter``.

The batch is executed twice inside one simulator session:
- run 1 (cold): includes simulator process launch + SHM setup.
- run 2 (warm): steady-state transport, comparable to the paper's per-stage
  measurements (Fig. 8). The reported phases come from the warm run.

Phase mapping onto vTen's pipeline (instrumented by wrapping the public
pipeline entry points from this script — no core-code changes):
- t_serialize : StreamSerializer.serialize (tensor -> wire-format bytes)
- t_compile   : RuntimeEngine.compile (IR lowering, includes t_serialize)
- t_load      : SHM image pack + write + doorbell (host -> sim data load)
- t_exec      : _wait_completion (simulator executes the batch)
- t_readback  : output SHM -> tensor deserialization
- t_verify    : golden forward() + bit-exact compare

Usage:
    python run_vten.py --n 4096 [--seed 42] [--json out.json]

Requires the repo root on PYTHONPATH (handled below) and a Verilator build
of the passthrough kernel (built automatically on first use).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from time import perf_counter

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
PROJECT = REPO_ROOT / "examples" / "passthrough"
KERNEL = "passthrough"

sys.path.insert(0, str(REPO_ROOT))


class PhaseTimer:
    """Accumulates wall-clock time per phase by wrapping methods."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}

    def reset(self) -> None:
        self.phases = {}

    def wrap(self, cls: type, method: str, phase: str) -> None:
        orig = getattr(cls, method)

        @functools.wraps(orig)
        def timed(*args, **kwargs):
            t0 = perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                self.phases[phase] = (
                    self.phases.get(phase, 0.0) + perf_counter() - t0
                )

        setattr(cls, method, timed)


def instrument() -> PhaseTimer:
    from vten.backend.sim.base import SimBackend
    from vten.runtime.context import ExecutionContext
    from vten.runtime.engine import RuntimeEngine
    from vten.runtime.serializer import StreamSerializer

    timer = PhaseTimer()
    timer.wrap(StreamSerializer, "serialize", "t_serialize")
    timer.wrap(RuntimeEngine, "compile", "t_compile")
    timer.wrap(SimBackend, "_pack_shm_image", "t_load")
    timer.wrap(SimBackend, "_submit_shm", "t_submit_cold")
    timer.wrap(SimBackend, "_submit_batch_internal_raw", "t_load")
    timer.wrap(SimBackend, "_wait_completion", "t_exec")
    timer.wrap(ExecutionContext, "_read_output_tensors", "t_readback")
    timer.wrap(ExecutionContext, "_auto_verify_all", "t_verify")
    return timer


def ensure_built() -> None:
    binary = PROJECT / "kernels" / KERNEL / "build" / "obj_dir" / "Vtb_top"
    if binary.exists():
        return
    from vten.cli.build import build_project

    print("Vtb_top missing - running vten build (one-time)...", file=sys.stderr)
    build_project(str(PROJECT), kernel_name=KERNEL, backend="verilator")


def run(n: int, seed: int = 42) -> dict:
    from vten.backend.base import RunContext
    from vten.backend.registry import get_backend
    from vten.cli.config import load_project_config
    from vten.cli.run import discover_kernel_class
    from vten.execution import execute_batch

    if n % 32 != 0:
        raise SystemExit("N must be a multiple of 32 (whole 256-bit beats)")

    config = load_project_config(PROJECT)
    # Bound host-side waits well under the benchmark's external kill timeout.
    config.setdefault("backend", {}).setdefault("verilator", {})[
        "submit_timeout_s"
    ] = 130

    kernel_dir = PROJECT / "kernels" / KERNEL
    kernel_cls = discover_kernel_class(KERNEL, kernel_dir)
    backend = get_backend("verilator", config)
    backend.set_run_context(RunContext(
        project_dir=PROJECT,
        kernel_build_dir=kernel_dir / "build",
    ))

    timer = instrument()
    runs: list[dict] = []
    prev_cwd = os.getcwd()
    os.chdir(backend.working_directory(kernel_dir, PROJECT))
    try:
        with backend:
            for label in ("cold", "warm"):
                timer.reset()
                t0 = perf_counter()
                batch = execute_batch(
                    backend=backend,
                    kernel_class=kernel_cls,
                    configs=[{"N": n, "seed": seed}],
                    verify=True,
                    project_dir=PROJECT,
                    on_error="raise",
                )
                t_total = perf_counter() - t0
                result = batch.single()
                assert result.verification_count >= 1, "no verification ran"
                stats = result.per_command_stats or []
                # Cycle counters accumulate across batches in one simulator
                # session; the batch's own span is commit(last) - issue(first).
                issues = [s.issue_cycle for s in stats if s.commit_cycle]
                commits = [s.commit_cycle for s in stats if s.commit_cycle]
                runs.append({
                    "label": label,
                    "t_total_s": t_total,
                    "cycles": max(commits, default=0) - min(issues, default=0),
                    "beats": sum(s.total_beats for s in stats),
                    **timer.phases,
                })
    finally:
        os.chdir(prev_cwd)
        backend.cleanup()

    warm = runs[-1]
    return {
        "framework": "vten",
        "n": n,
        "beats": warm["beats"] // 2,  # PUSH + PULL each count the beats once
        "cycles": warm["cycles"],
        "t_serialize_s": warm.get("t_serialize", 0.0),
        "t_compile_s": warm.get("t_compile", 0.0),
        "t_load_s": warm.get("t_load", 0.0),
        "t_exec_s": warm.get("t_exec", 0.0),
        "t_readback_s": warm.get("t_readback", 0.0),
        "t_verify_s": warm.get("t_verify", 0.0),
        "t_total_s": warm["t_total_s"],
        "t_total_cold_s": runs[0]["t_total_s"],
        "verified": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1024, help="tensor elements (int8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, help="write result JSON here")
    args = ap.parse_args()

    ensure_built()
    result = run(args.n, args.seed)

    out = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(out)
    print(out)


if __name__ == "__main__":
    main()
