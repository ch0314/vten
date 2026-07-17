"""Benchmark wrapper around the hand-written Cocotb testbench.

Reuses the functional testbench primitives from ``test_passthrough`` and adds
wall-clock phase instrumentation (`time.perf_counter`) plus a JSON report.
Kept separate from ``test_passthrough.py`` so the verification-LOC count of
the plain testbench is not inflated by benchmark-only code.

Phases (mirroring the paper's Fig. 8 stage decomposition):
- t_serialize : host tensor -> 256-bit beat words (wire format)
- t_exec      : streaming the beats through the simulator (the per-beat
                Python/VPI round-trips; Cocotb has no bulk data-load path,
                so data loading is folded into execution by construction)
- t_check     : beat words -> tensor + golden compare
"""

import json
import os
from time import perf_counter

import numpy as np

import cocotb
from cocotb.triggers import RisingEdge
from cocotb.utils import get_sim_time

from test_passthrough import (
    AxisSink,
    AxisSource,
    pack_beats,
    setup_dut,
    unpack_beats,
)


@cocotb.test(timeout_time=120, timeout_unit="sec")
async def bench_passthrough(dut):
    n = int(os.environ.get("BENCH_N", 1024))
    seed = int(os.environ.get("BENCH_SEED", 42))
    report_path = os.environ.get("BENCH_RESULT_JSON", "")

    rng = np.random.default_rng(seed)
    tensor = rng.integers(-128, 128, size=n, dtype=np.int8)
    golden = tensor.copy()

    await setup_dut(dut)

    t0 = perf_counter()
    beats = pack_beats(tensor)
    t1 = perf_counter()

    source = AxisSource(dut)
    sink = AxisSink(dut)
    await RisingEdge(dut.clk)  # let tready settle before timing starts
    sim_t0 = get_sim_time("ns")
    t2 = perf_counter()
    send_task = cocotb.start_soon(source.send(beats))
    received = await sink.recv(len(beats))
    await send_task
    t3 = perf_counter()
    sim_t1 = get_sim_time("ns")

    result = unpack_beats(received, tensor.size)
    assert np.array_equal(result, golden), "DUT output != golden"
    t4 = perf_counter()

    report = {
        "n": n,
        "beats": len(beats),
        "cycles": int((sim_t1 - sim_t0) / 10),  # 10 ns clock period
        "t_serialize_s": t1 - t0,
        "t_exec_s": t3 - t2,
        "t_check_s": t4 - t3,
        "verified": True,
    }
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
    dut._log.info("bench report: %s", json.dumps(report))
