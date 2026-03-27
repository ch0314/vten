"""Integration tests for vten.functional — run_kernel & KernelExecutor.

Unlike test_functional.py (mock-based), these tests exercise the real
compile pipeline (Stages 0-7) without a simulator backend.

Tests verify:
1. run_kernel: full compile → SHM image generation (no backend = compile-only)
2. KernelExecutor: repeated calls with alias detection in compiled IR
3. configure=True: auto_bind register commands in IR
4. Multi-input kernels: correct SEND/RECV and buffer allocation
"""

from __future__ import annotations

import struct

import pytest
import torch

from vten.functional import KernelExecutor, run_kernel
from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.runtime.context import BatchResult, ExecutionContext
from vten.spec.models import (
    AutoBindSpec,
    Direction,
    InterfaceSpec,
    KernelSpec,
    MemoryRegion,
    OpCode,
    OpKind,
    PackingScheme,
    Protocol,
    RegisterSpec,
    Role,
)


# ── Test Kernel Definitions ─────────────────────────────────────


class StreamKernel(Kernel):
    """Minimal streaming kernel for integration tests."""

    data_in = Tensor(shape=(32,), dtype=torch.int8, interface="axis_in")
    data_out = Tensor(shape=(32,), dtype=torch.int8, interface="axis_out")

    def generate_inputs(self, seed=None):
        self.data_in.fill_random()

    def forward(self):
        return self.data_in.data.clone()


class TwoInputStreamKernel(Kernel):
    """Two inputs, one output."""

    ifm = Tensor(shape=(32,), dtype=torch.int8, interface="ifm_port")
    wgt = Tensor(shape=(32,), dtype=torch.int8, interface="wgt_port")
    ofm = Tensor(shape=(32,), dtype=torch.int8, interface="ofm_port")

    def forward(self):
        return (self.ifm.data.to(torch.int16) + self.wgt.data.to(torch.int16)).clamp(-128, 127).to(torch.int8)


class RegKernel(Kernel):
    """Kernel with AXI-Lite registers for configure() test."""

    data_in = Tensor(shape=(32,), dtype=torch.int8, interface="ddr",
                     direction=Direction.HOST_TO_DEV)
    data_out = Tensor(shape=(32,), dtype=torch.int8, interface="ddr",
                      direction=Direction.DEV_TO_HOST)
    ctrl = register("ctrl")


# ── Spec Helpers ─────────────────────────────────────────────────


def _stream_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="stream_test",
        rtl_top="rtl/stream.sv",
        interfaces={
            "axis_in": InterfaceSpec(
                name="axis_in",
                rtl_port="s_axis_in",
                protocol=Protocol.AXI4S,
                tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axis_out": InterfaceSpec(
                name="axis_out",
                rtl_port="m_axis_out",
                protocol=Protocol.AXI4S,
                tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


def _two_input_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="two_input_test",
        rtl_top="rtl/two_input.sv",
        interfaces={
            "ifm_port": InterfaceSpec(
                name="ifm_port",
                rtl_port="s_axis_ifm",
                protocol=Protocol.AXI4S,
                tensor="ifm",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "wgt_port": InterfaceSpec(
                name="wgt_port",
                rtl_port="s_axis_wgt",
                protocol=Protocol.AXI4S,
                tensor="wgt",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "ofm_port": InterfaceSpec(
                name="ofm_port",
                rtl_port="m_axis_ofm",
                protocol=Protocol.AXI4S,
                tensor="ofm",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


def _reg_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="reg_test",
        rtl_top="rtl/reg.sv",
        memory_regions={
            "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000, alignment=64),
        },
        interfaces={
            "ctrl": InterfaceSpec(
                name="ctrl",
                rtl_port="s_axilite_ctrl",
                protocol=Protocol.AXI4L,
                addr_width=16,
                registers=[
                    RegisterSpec(
                        name="in_addr_lo",
                        offset=0x14,
                        auto_bind=AutoBindSpec(tensor="data_in", value="address", bits="31:0"),
                    ),
                    RegisterSpec(
                        name="in_addr_hi",
                        offset=0x18,
                        auto_bind=AutoBindSpec(tensor="data_in", value="address", bits="63:32"),
                    ),
                    RegisterSpec(
                        name="length",
                        offset=0x1C,
                        auto_bind=AutoBindSpec(tensor="data_in", value="size_beats"),
                    ),
                ],
            ),
            "ddr": InterfaceSpec(
                name="ddr",
                rtl_port="m_axi_ddr",
                protocol=Protocol.AXI4,
                data_width=256,
                addr_width=64,
                memory_region="ddr",
                tensors=["data_in", "data_out"],
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# run_kernel integration tests
# ═══════════════════════════════════════════════════════════════════


class TestRunKernelCompile:
    """run_kernel with no backend: exercises full compile pipeline."""

    def test_compile_produces_shm_image(self):
        """Compile-only path produces non-empty SHM image."""
        # Capture CompiledResult via patching the backend submit
        compiled_results = []
        original_run = ExecutionContext.run

        def capture_compile_run(self):
            # Run compile but not backend submit
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(StreamKernel, {"data_in": x}, spec=_stream_spec())

        cr = compiled_results[0]
        assert len(cr.shm_image) > 0
        # Check SHM magic
        magic = struct.unpack_from("<I", cr.shm_image, 0)[0]
        assert magic == 0x5654454E  # "VTEN"

    def test_compile_produces_commands(self):
        """Compile produces LOAD, PUSH, PULL, STORE commands."""
        compiled_results = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(StreamKernel, {"data_in": x}, spec=_stream_spec())

        commands = compiled_results[0].commands
        opcodes = {cmd.op for cmd in commands}
        # send_tensor = LOAD + PUSH; recv_tensor = PULL (no STORE for AXI4S)
        assert OpCode.LOAD in opcodes
        assert OpCode.PUSH in opcodes
        assert OpCode.PULL in opcodes

    def test_multiple_inputs_buffer_allocation(self):
        """Two inputs + one output: 3 distinct buffer_ids."""
        compiled_results = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        ifm = torch.randint(-128, 127, (32,), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(TwoInputStreamKernel, {"ifm": ifm, "wgt": wgt}, spec=_two_input_spec())

        buffer_ids = compiled_results[0].buffer_ids
        assert len(buffer_ids) == 3  # ifm, wgt, ofm
        assert len(set(buffer_ids.values())) == 3  # all unique

    def test_configure_generates_write_reg(self):
        """configure=True generates WRITE_REG commands for auto_bind registers."""
        compiled_results = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(RegKernel, {"data_in": x}, configure=True, spec=_reg_spec())

        commands = compiled_results[0].commands
        write_regs = [c for c in commands if c.op == OpCode.WRITE_REG]
        # 3 auto_bind registers: in_addr_lo, in_addr_hi, length
        assert len(write_regs) >= 3


# ═══════════════════════════════════════════════════════════════════
# KernelExecutor integration tests
# ═══════════════════════════════════════════════════════════════════


class TestKernelExecutorCompile:
    """KernelExecutor compile-only integration tests."""

    def test_repeated_calls_compile(self):
        """Two consecutive calls each compile independently."""
        compile_count = [0]

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            engine.compile(target="sim")
            compile_count[0] += 1
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            npu = KernelExecutor(StreamKernel, spec=_stream_spec())
            npu(data_in=x)
            npu(data_in=x)

        assert compile_count[0] == 2

    def test_alias_skips_load_in_ir(self):
        """When alias is triggered, second call should have fewer LOAD commands."""
        all_commands = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            all_commands.append(result.commands)
            return BatchResult(
                status="DONE",
                output_tensors={"data_out": torch.zeros(32, dtype=torch.int8)},
            )

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            npu = KernelExecutor(StreamKernel, spec=_stream_spec())

            # First call: normal
            result1 = npu(data_in=x)

            # Second call: pass output as input → alias
            y = result1["data_out"]
            result2 = npu(data_in=y)

        # First call: should have LOAD for data_in
        first_loads = [c for c in all_commands[0] if c.op == OpCode.LOAD]
        # Second call: LOAD for data_in should be skipped (alias)
        second_loads = [c for c in all_commands[1] if c.op == OpCode.LOAD]
        assert len(second_loads) < len(first_loads)

    def test_executor_two_inputs_alias_partial(self):
        """With two inputs, only the aliased one skips LOAD."""
        all_commands = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            all_commands.append(result.commands)
            return BatchResult(
                status="DONE",
                output_tensors={"ofm": torch.zeros(32, dtype=torch.int8)},
            )

        from unittest.mock import patch
        ifm = torch.randint(-128, 127, (32,), dtype=torch.int8)
        wgt = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            npu = KernelExecutor(TwoInputStreamKernel, spec=_two_input_spec())

            # First call: 2 LOADs (ifm + wgt)
            result1 = npu(ifm=ifm, wgt=wgt)

            # Second call: pass ofm output as ifm → alias ifm, fresh wgt
            ofm_output = result1["ofm"]
            npu(ifm=ofm_output, wgt=wgt)

        first_loads = [c for c in all_commands[0] if c.op == OpCode.LOAD]
        second_loads = [c for c in all_commands[1] if c.op == OpCode.LOAD]
        # First: 2 LOADs; Second: 1 LOAD (only wgt, ifm aliased)
        assert len(first_loads) == 2
        assert len(second_loads) == 1


# ═══════════════════════════════════════════════════════════════════
# SHM Image verification
# ═══════════════════════════════════════════════════════════════════


class TestSHMImageIntegrity:
    """Verify SHM image structure from functional API."""

    def test_shm_header_fields(self):
        """SHM header has correct magic, version, command/buffer counts."""
        compiled_results = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.randint(-128, 127, (32,), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(StreamKernel, {"data_in": x}, spec=_stream_spec())

        shm = compiled_results[0].shm_image
        magic, version = struct.unpack_from("<II", shm, 0)
        # SHM header layout: 0x08=host_status, 0x10=num_cmds, 0x14=num_bufs
        num_cmds = struct.unpack_from("<I", shm, 0x10)[0]
        num_bufs = struct.unpack_from("<I", shm, 0x14)[0]

        assert magic == 0x5654454E
        assert version == 0x00000003
        assert num_cmds > 0
        assert num_bufs > 0

    def test_shm_data_contains_tensor(self):
        """SHM data region contains the input tensor bytes."""
        compiled_results = []

        def capture_compile_run(self):
            from vten.runtime.engine import RuntimeEngine
            engine = RuntimeEngine(
                self._kernels, self._pending_ops,
                self._project_params, self._alias_registry,
            )
            result = engine.compile(target="sim")
            compiled_results.append(result)
            return BatchResult(status="DONE")

        from unittest.mock import patch
        x = torch.tensor(list(range(32)), dtype=torch.int8)

        with patch.object(ExecutionContext, "run", capture_compile_run):
            run_kernel(StreamKernel, {"data_in": x}, spec=_stream_spec())

        shm = compiled_results[0].shm_image
        # The serialized tensor data should appear somewhere in the SHM image
        expected_bytes = bytes(range(32))
        assert expected_bytes in shm
