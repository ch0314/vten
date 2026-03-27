"""Tests for single-batch multi-config compilation.

Verifies:
1. IRLowering cmd_id/buffer_id offsets
2. RuntimeEngine.compile_multi() merges commands with BARRIER
3. ExecutionContext.config_boundary() + run() multi-config path
"""

from __future__ import annotations

import struct

import pytest
import torch

from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.context import BatchResult, ExecutionContext
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    OpCode,
    PackingScheme,
    Protocol,
)


# ── Test Kernel ──


class StreamKernel(Kernel):
    data_in = Tensor(shape=("${N}",), dtype=torch.int8, interface="axis_in")
    data_out = Tensor(shape=("${N}",), dtype=torch.int8, interface="axis_out")

    def forward(self):
        return self.data_in.data.clone()


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


# ── IRLowering offset tests ──


class TestIRLoweringOffsets:

    def test_cmd_id_offset(self):
        """Commands start from cmd_id_start, not 0."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.context import AliasRegistry

        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=spec, N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        engine = RuntimeEngine(
            kernels=ctx._kernels,
            ops=ctx._pending_ops,
            project_params={"N": 32},
            alias_registry=ctx._alias_registry,
        )

        # Record ops
        ctx.send_tensor(ki.get_tensor("data_in"))
        ctx.recv_tensor(ki.get_tensor("data_out"))

        engine2 = RuntimeEngine(
            kernels=ctx._kernels,
            ops=ctx._pending_ops,
            project_params={"N": 32},
            alias_registry=ctx._alias_registry,
        )

        view, commands, buffer_ids, _, _ = engine2._compile_ir(
            cmd_id_start=10, buffer_id_start=5,
        )

        assert all(c.cmd_id >= 10 for c in commands)
        assert all(v >= 5 for v in buffer_ids.values())

    def test_buffer_id_offset(self):
        """Buffer IDs start from buffer_id_start."""
        from vten.runtime.ir import IRLowering
        from vten.runtime.flattener import KernelInstance

        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": 32})
        ki = ctx.instantiate(StreamKernel, spec=spec, N=32)
        ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)

        ctx.send_tensor(ki.get_tensor("data_in"))
        ctx.recv_tensor(ki.get_tensor("data_out"))

        from vten.runtime.engine import RuntimeEngine

        engine = RuntimeEngine(
            kernels=ctx._kernels,
            ops=ctx._pending_ops,
            project_params={"N": 32},
            alias_registry=ctx._alias_registry,
        )

        view, commands, buffer_ids, _, _ = engine._compile_ir(
            buffer_id_start=100,
        )

        assert min(buffer_ids.values()) >= 100


# ── compile_multi tests ──


class TestCompileMulti:

    def _make_engine(self, N=32):
        """Create an engine with send+recv ops for a StreamKernel."""
        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": N})
        ki = ctx.instantiate(StreamKernel, spec=spec, N=N)
        ki.get_tensor("data_in").data = torch.randint(
            -128, 127, (N,), dtype=torch.int8,
        )
        ctx.send_tensor(ki.get_tensor("data_in"))
        ctx.recv_tensor(ki.get_tensor("data_out"))

        from vten.runtime.engine import RuntimeEngine

        return RuntimeEngine(
            kernels=ctx._kernels,
            ops=ctx._pending_ops,
            project_params={"N": N},
            alias_registry=ctx._alias_registry,
        )

    def test_single_engine_passthrough(self):
        """compile_multi with 1 engine delegates to compile()."""
        from vten.runtime.engine import RuntimeEngine

        engine = self._make_engine()
        result = RuntimeEngine.compile_multi([engine])
        assert len(result.commands) > 0
        assert result.shm_image

    def test_two_configs_barrier_insertion(self):
        """Two config groups produce commands with a BARRIER between them."""
        from vten.runtime.engine import RuntimeEngine

        e1 = self._make_engine(N=32)
        e2 = self._make_engine(N=64)
        result = RuntimeEngine.compile_multi([e1, e2])

        opcodes = [c.op for c in result.commands]
        assert OpCode.BARRIER in opcodes, "BARRIER must be inserted between groups"

        # BARRIER should not be the first or last command
        barrier_idx = opcodes.index(OpCode.BARRIER)
        assert barrier_idx > 0
        assert barrier_idx < len(opcodes) - 1

    def test_cmd_ids_globally_unique(self):
        """All cmd_ids across groups are unique and sequential."""
        from vten.runtime.engine import RuntimeEngine

        e1 = self._make_engine(N=32)
        e2 = self._make_engine(N=32)
        e3 = self._make_engine(N=32)
        result = RuntimeEngine.compile_multi([e1, e2, e3])

        cmd_ids = [c.cmd_id for c in result.commands]
        assert len(cmd_ids) == len(set(cmd_ids)), "cmd_ids must be unique"
        # Should be monotonically increasing
        assert cmd_ids == sorted(cmd_ids)

    def test_buffer_ids_per_group_unique(self):
        """Buffer IDs from different groups don't overlap."""
        from vten.runtime.engine import RuntimeEngine

        e1 = self._make_engine(N=32)
        e2 = self._make_engine(N=32)
        result = RuntimeEngine.compile_multi([e1, e2])

        # Extract buffer_ids for each config group
        cfg0_ids = {
            v for k, v in result.buffer_ids.items()
            if k.startswith("cfg0:")
        }
        cfg1_ids = {
            v for k, v in result.buffer_ids.items()
            if k.startswith("cfg1:")
        }
        assert cfg0_ids.isdisjoint(cfg1_ids), \
            f"Buffer IDs overlap: {cfg0_ids & cfg1_ids}"

    def test_shm_header_valid(self):
        """Merged SHM image has valid header."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.shm import SHM_MAGIC, PROTOCOL_VERSION

        e1 = self._make_engine(N=32)
        e2 = self._make_engine(N=32)
        result = RuntimeEngine.compile_multi([e1, e2])

        shm = result.shm_image
        magic = struct.unpack_from("<I", shm, 0x00)[0]
        version = struct.unpack_from("<I", shm, 0x04)[0]
        num_cmds = struct.unpack_from("<I", shm, 0x10)[0]
        num_bufs = struct.unpack_from("<I", shm, 0x14)[0]

        assert magic == SHM_MAGIC
        assert version == PROTOCOL_VERSION
        assert num_cmds == len(result.commands)
        assert num_bufs > 0

    def test_three_configs_two_barriers(self):
        """Three config groups produce exactly two BARRIER commands."""
        from vten.runtime.engine import RuntimeEngine

        engines = [self._make_engine(N=32) for _ in range(3)]
        result = RuntimeEngine.compile_multi(engines)

        barrier_count = sum(1 for c in result.commands if c.op == OpCode.BARRIER)
        assert barrier_count == 2


# ── ExecutionContext.config_boundary() tests ──


class TestConfigBoundary:

    def test_config_boundary_splits_ops(self):
        """config_boundary() records boundary and resets kernels."""
        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": 32})

        ki1 = ctx.instantiate(StreamKernel, spec=spec, N=32)
        ki1.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
        ctx.send_tensor(ki1.get_tensor("data_in"))
        ctx.recv_tensor(ki1.get_tensor("data_out"))

        ops_before = len(ctx._pending_ops)
        ctx.config_boundary()

        # Boundary recorded
        assert len(ctx._config_boundaries) == 1
        assert ctx._config_boundaries[0] == ops_before
        # Kernels snapshot saved
        assert len(ctx._config_kernels) == 1
        assert "StreamKernel" in ctx._config_kernels[0]
        # Current kernels reset
        assert len(ctx._kernels) == 0

    def test_multi_config_compile_produces_barrier(self):
        """run() with config_boundary() produces commands with BARRIER."""
        compiled_results = []

        def capture_run(self_ctx):
            from vten.runtime.engine import RuntimeEngine

            if self_ctx._config_boundaries:
                compiled = self_ctx._compile_multi_config("sim")
            else:
                engine = RuntimeEngine(
                    kernels=self_ctx._kernels,
                    ops=self_ctx._pending_ops,
                    project_params=self_ctx._project_params,
                    alias_registry=self_ctx._alias_registry,
                )
                compiled = engine.compile(target="sim")
            compiled_results.append(compiled)
            return BatchResult(status="DONE")

        from unittest.mock import patch

        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": 32})

        # Config 1
        ki1 = ctx.instantiate(StreamKernel, spec=spec, N=32)
        ki1.get_tensor("data_in").data = torch.randint(-128, 127, (32,), dtype=torch.int8)
        ctx.send_tensor(ki1.get_tensor("data_in"))
        ctx.recv_tensor(ki1.get_tensor("data_out"))
        ctx.config_boundary()

        # Config 2
        ki2 = ctx.instantiate(StreamKernel, spec=spec, N=64)
        ki2.get_tensor("data_in").data = torch.randint(-128, 127, (64,), dtype=torch.int8)
        ctx.send_tensor(ki2.get_tensor("data_in"))
        ctx.recv_tensor(ki2.get_tensor("data_out"))

        with patch.object(ExecutionContext, "run", capture_run):
            ctx.run()

        cr = compiled_results[0]
        opcodes = [c.op for c in cr.commands]
        assert OpCode.BARRIER in opcodes

    def test_multi_config_cmd_ids_unique(self):
        """Multi-config through config_boundary produces unique cmd_ids."""
        compiled_results = []

        def capture_run(self_ctx):
            from vten.runtime.engine import RuntimeEngine

            if self_ctx._config_boundaries:
                compiled = self_ctx._compile_multi_config("sim")
            else:
                engine = RuntimeEngine(
                    kernels=self_ctx._kernels,
                    ops=self_ctx._pending_ops,
                    project_params=self_ctx._project_params,
                    alias_registry=self_ctx._alias_registry,
                )
                compiled = engine.compile(target="sim")
            compiled_results.append(compiled)
            return BatchResult(status="DONE")

        from unittest.mock import patch

        spec = _stream_spec()
        ctx = ExecutionContext(project_params={"N": 32})

        for _ in range(3):
            ki = ctx.instantiate(StreamKernel, spec=spec, N=32)
            ki.get_tensor("data_in").data = torch.zeros(32, dtype=torch.int8)
            ctx.send_tensor(ki.get_tensor("data_in"))
            ctx.recv_tensor(ki.get_tensor("data_out"))
            ctx.config_boundary()

        with patch.object(ExecutionContext, "run", capture_run):
            ctx.run()

        cr = compiled_results[0]
        cmd_ids = [c.cmd_id for c in cr.commands]
        assert len(cmd_ids) == len(set(cmd_ids)), "cmd_ids must be unique"
        assert cmd_ids == sorted(cmd_ids), "cmd_ids must be monotonically increasing"
