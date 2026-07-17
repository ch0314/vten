"""Hand-written Cocotb testbench for the `passthrough` AXI4-Stream DUT.

Drives a random int8 tensor into `s_axis`, collects `m_axis`, and compares
against the golden model (identity). This file is the verification-LOC
baseline for the vTen-vs-Cocotb comparison: it contains only the functional
testbench (drivers, packing, golden compare) — no benchmark instrumentation.
"""

import os

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

BYTES_PER_BEAT = 32  # 256-bit datapath, 32 x int8 elements per beat


def pack_beats(tensor: np.ndarray) -> list[int]:
    """Serialize an int8 tensor into 256-bit little-endian beat words."""
    raw = tensor.astype(np.int8).tobytes()
    assert len(raw) % BYTES_PER_BEAT == 0, "tensor must fill whole beats"
    return [
        int.from_bytes(raw[i:i + BYTES_PER_BEAT], "little")
        for i in range(0, len(raw), BYTES_PER_BEAT)
    ]


def unpack_beats(beats: list[int], num_elements: int) -> np.ndarray:
    """Deserialize 256-bit beat words back into an int8 tensor."""
    raw = b"".join(b.to_bytes(BYTES_PER_BEAT, "little") for b in beats)
    return np.frombuffer(raw[:num_elements], dtype=np.int8)


class AxisSource:
    """AXI4-Stream master: drives one beat per clock while tready is high."""

    def __init__(self, dut):
        self.clk = dut.clk
        self.tdata = dut.s_axis_tdata
        self.tvalid = dut.s_axis_tvalid
        self.tready = dut.s_axis_tready
        self.tlast = dut.s_axis_tlast
        self.tvalid.value = 0
        self.tlast.value = 0

    async def send(self, beats: list[int]) -> None:
        last = len(beats) - 1
        for i, beat in enumerate(beats):
            self.tdata.value = beat
            self.tvalid.value = 1
            self.tlast.value = int(i == last)
            while True:
                await ReadOnly()
                accepted = bool(self.tready.value)
                await RisingEdge(self.clk)
                if accepted:
                    break
        self.tvalid.value = 0
        self.tlast.value = 0


class AxisSink:
    """AXI4-Stream slave: holds tready high and samples beats at each edge."""

    def __init__(self, dut):
        self.clk = dut.clk
        self.tdata = dut.m_axis_tdata
        self.tvalid = dut.m_axis_tvalid
        self.tready = dut.m_axis_tready
        self.tlast = dut.m_axis_tlast
        self.tready.value = 1
        self.cycles = 0

    async def recv(self, num_beats: int) -> list[int]:
        beats = []
        saw_last = False
        while len(beats) < num_beats:
            await ReadOnly()
            if self.tvalid.value and self.tready.value:
                beats.append(int(self.tdata.value))
                saw_last = bool(self.tlast.value)
            await RisingEdge(self.clk)
            self.cycles += 1
        assert saw_last, "tlast not asserted on final beat"
        return beats


async def stream_tensor(dut, tensor: np.ndarray) -> np.ndarray:
    """Push `tensor` through the DUT and return the received tensor."""
    beats = pack_beats(tensor)
    source = AxisSource(dut)
    sink = AxisSink(dut)
    send_task = cocotb.start_soon(source.send(beats))
    received = await sink.recv(len(beats))
    await send_task
    return unpack_beats(received, tensor.size)


async def setup_dut(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.m_axis_tready.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test(timeout_time=120, timeout_unit="sec")
async def test_passthrough(dut):
    """Random int8 tensor in, bit-exact identity out."""
    n = int(os.environ.get("BENCH_N", 1024))
    seed = int(os.environ.get("BENCH_SEED", 42))
    rng = np.random.default_rng(seed)
    tensor = rng.integers(-128, 128, size=n, dtype=np.int8)
    golden = tensor.copy()

    await setup_dut(dut)
    result = await stream_tensor(dut, tensor)

    assert np.array_equal(result, golden), "DUT output != golden"
