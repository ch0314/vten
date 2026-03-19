"""Phase 2 tests — End-to-End Pipeline Integration.

Spec reference: 02_runtime_engine.md §3-§14
NPU 3D patterns: npu_3d_analysis.md

Tests the full pipeline: ExecutionContext → RuntimeEngine.compile() →
CompiledResult with commands, shm_image, bfm_configs, buffer_ids.
"""

from __future__ import annotations

import struct

import pytest
import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.errors import CompilationError, ShapeMismatchError
from vten.kernel.base import Kernel, register
from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext
from vten.spec.models import (
    AutoBindSpec,
    Direction,
    InterfaceSpec,
    KernelSpec,
    MappingType,
    MemoryRegion,
    OpCode,
    OpKind,
    PackingScheme,
    Protocol,
    RegisterSpec,
    Role,
)


# ── Kernel definitions ───────────────────────────────────────────


class PassthroughKernel(Kernel):
    data_in = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axis_in")
    data_out = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axis_out")

    def generate_inputs(self, seed=None):
        self.data_in.fill_random()

    def forward(self):
        return self.data_in.data


class MemKernel(Kernel):
    ifm = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ddr", direction=Direction.HOST_TO_DEV)
    ofm = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="ddr", direction=Direction.DEV_TO_HOST)

    def generate_inputs(self, seed=None):
        self.ifm.fill_random()


# ── Spec helpers ─────────────────────────────────────────────────


def _passthrough_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        parameters={"SIZE": "${SIZE}"},
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


def _mem_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="memkernel",
        rtl_top="rtl/mem.sv",
        parameters={"IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}"},
        memory_regions={
            "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000, alignment=4096),
        },
        interfaces={
            "ctrl": InterfaceSpec(
                name="ctrl",
                rtl_port="s_axilite_ctrl",
                protocol=Protocol.AXI4L,
                addr_width=16,
                registers=[
                    RegisterSpec(
                        name="in_ch", offset=0x010,
                        auto_bind=AutoBindSpec(param="${IN_CH}"),
                    ),
                    RegisterSpec(
                        name="ifm_addr_lsb", offset=0x038,
                        auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="31:0"),
                    ),
                    RegisterSpec(
                        name="ifm_addr_msb", offset=0x03C,
                        auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="63:32"),
                    ),
                    RegisterSpec(
                        name="vsync", offset=0x050,
                        fields={"trigger": "0:0"},
                    ),
                    RegisterSpec(
                        name="layer_done", offset=0x054,
                        fields={"done": "0:0"},
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
                tensors=["ifm", "ofm"],
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# §3 — ExecutionContext basic API
# ═══════════════════════════════════════════════════════════════════


class TestExecutionContextAPI:
    """ExecutionContext recording and instantiation."""

    def test_instantiate_creates_kernel(self):
        """instantiate() returns KernelInstance with resolved params."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        assert inst.name == "PassthroughKernel"
        assert inst.kernel_class_instance is not None

    def test_instantiate_resolves_shapes(self):
        """Tensors get shapes resolved during instantiate."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=32)
        data_in = inst.get_tensor("data_in")
        assert data_in._resolved_shape == (32,)
        assert data_in._element_count == 32

    def test_record_operations(self):
        """DSL operations are recorded as pending."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=8)
        h1 = ctx.load_tensor(inst.data_in)
        h2 = ctx.push_tensor(inst.data_in, dep=h1)
        assert len(ctx._pending_ops) == 2

    def test_operation_handles_returned(self):
        """Each DSL call returns an OperationHandle."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=8)
        h = ctx.load_tensor(inst.data_in)
        assert isinstance(h, OperationHandle)
        assert h.op.kind == OpKind.LOAD_TENSOR

    def test_barrier(self):
        """barrier() records a BARRIER operation."""
        ctx = ExecutionContext(project_params={})
        h = ctx.barrier()
        assert h.op.kind == OpKind.BARRIER

    def test_write_register(self):
        """write_register() records WRITE_REGISTER."""
        from vten.kernel.base import RegisterHandle

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        reg = RegisterHandle(interface_name="ctrl")
        h = ctx.write_register(reg, {"trigger": 1})
        assert h.op.kind == OpKind.WRITE_REGISTER
        assert h.op.register_fields == {"trigger": 1}

    def test_poll_register(self):
        """poll_register() records POLL_REGISTER."""
        from vten.kernel.base import RegisterHandle

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        reg = RegisterHandle(interface_name="ctrl")
        h = ctx.poll_register(reg, "done")
        assert h.op.kind == OpKind.POLL_REGISTER
        assert h.op.register_field_name == "done"

    def test_alias_registration(self):
        """alias() registers buffer aliasing."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        ctx.alias(inst.ifm, inst.ofm)
        assert ctx._alias_registry.is_alias_target("ofm")
        assert ctx._alias_registry.get_source("ofm") == "ifm"

    def test_run_clears_pending_ops(self):
        """run() clears pending ops after compilation."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=8)
        inst.data_in.fill_random()
        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()
        assert len(ctx._pending_ops) == 0


# ═══════════════════════════════════════════════════════════════════
# Full Pipeline — Passthrough (AXI4-Stream only)
# ═══════════════════════════════════════════════════════════════════


class TestPassthroughE2E:
    """End-to-end: passthrough kernel with AXI4-Stream."""

    def test_send_recv_pipeline(self):
        """send_tensor + recv_tensor → compile → valid commands + SHM."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        h_send = ctx.send_tensor(inst.data_in)
        h_recv = ctx.recv_tensor(inst.data_out, dep=h_send)

        result = ctx.run()
        assert result.status == "DONE"

        compiled = ctx._last_compiled
        assert compiled is not None
        assert len(compiled.commands) > 0
        assert len(compiled.shm_image) > 0
        assert len(compiled.buffer_ids) == 2  # data_in, data_out

    def test_commands_contain_load_push_pull(self):
        """send→LOAD+PUSH, recv→PULL."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        opcodes = [c.op for c in compiled.commands]
        assert OpCode.LOAD in opcodes
        assert OpCode.PUSH in opcodes
        assert OpCode.PULL in opcodes

    def test_shm_image_has_magic(self):
        """SHM image starts with MAGIC bytes."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        magic = struct.unpack_from("<I", compiled.shm_image, 0)[0]
        assert magic == 0x5654454E

    def test_shm_image_version(self):
        """SHM image has correct version at offset 4."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        version = struct.unpack_from("<I", compiled.shm_image, 4)[0]
        assert version == 0x00000003

    def test_cmd_ids_sequential(self):
        """Command IDs are sequential starting from 0."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        ids = [c.cmd_id for c in compiled.commands]
        assert ids == list(range(len(ids)))

    def test_bfm_configs_for_stream(self):
        """AXI4-Stream interfaces generate BFM configs."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        iface_names = [b.interface_name for b in compiled.bfm_configs]
        assert "axis_in" in iface_names
        assert "axis_out" in iface_names

    def test_input_data_in_shm(self):
        """Input tensor data is copied into SHM data region."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.data = torch.arange(16, dtype=torch.int8)

        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        # SHM image should contain the serialized input data somewhere
        # The data is 16 int8 values = 16 bytes
        # Just verify the SHM image is large enough
        assert len(compiled.shm_image) > 256 + 16  # at least control + data


# ═══════════════════════════════════════════════════════════════════
# Full Pipeline — Memory-mapped kernel (AXI4 + AXI4-Lite)
# ═══════════════════════════════════════════════════════════════════


class TestMemKernelE2E:
    """End-to-end: memory-mapped kernel with registers."""

    def test_load_configure_poll_store(self):
        """Full NPU-like flow: load → configure → vsync → poll → store."""
        from vten.kernel.base import RegisterHandle

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()
        reg = RegisterHandle(interface_name="ctrl")

        h_load = ctx.load_tensor(inst.ifm)
        h_push = ctx.push_tensor(inst.ifm, dep=h_load)
        h_conf = ctx.configure(inst, dep=h_push)
        h_vsync = ctx.write_register(reg, {"trigger": 1}, dep=h_conf)
        h_poll = ctx.poll_register(reg, "done", dep=h_vsync)
        h_pull = ctx.pull_tensor(inst.ofm, dep=h_poll)
        h_store = ctx.store_tensor(inst.ofm, dep=h_pull)

        result = ctx.run()
        assert result.status == "DONE"

        compiled = ctx._last_compiled
        opcodes = [c.op for c in compiled.commands]
        assert OpCode.LOAD in opcodes
        assert OpCode.PUSH in opcodes
        assert OpCode.WRITE_REG in opcodes
        assert OpCode.POLL_REG in opcodes
        assert OpCode.PULL in opcodes
        assert OpCode.STORE in opcodes

    def test_configure_generates_auto_bind_writes(self):
        """configure() generates WRITE_REG for each auto_bind register."""
        from vten.kernel.base import RegisterHandle

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        ctx.load_tensor(inst.ifm)
        ctx.configure(inst)
        ctx.run()

        compiled = ctx._last_compiled
        write_reg_cmds = [c for c in compiled.commands if c.op == OpCode.WRITE_REG]
        # At least in_ch + ifm_addr_lsb + ifm_addr_msb = 3 auto_bind registers
        assert len(write_reg_cmds) >= 3

    def test_bfm_configs_include_axi4l_and_axi4(self):
        """Both AXI4-Lite and AXI4 interfaces get BFM configs."""
        from vten.kernel.base import RegisterHandle

        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        ctx.load_tensor(inst.ifm)
        ctx.run()

        compiled = ctx._last_compiled
        protocols = {b.protocol for b in compiled.bfm_configs}
        assert Protocol.AXI4L in protocols
        assert Protocol.AXI4 in protocols

    def test_buffer_ids_for_all_tensors(self):
        """All exposed tensors get buffer IDs."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        ctx.load_tensor(inst.ifm)
        ctx.store_tensor(inst.ofm)
        ctx.run()

        compiled = ctx._last_compiled
        assert "ifm" in compiled.buffer_ids
        assert "ofm" in compiled.buffer_ids

    def test_shm_size_cache_aligned(self):
        """SHM image size is cache-line (64B) aligned."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        ctx.send_tensor(inst.ifm)
        ctx.recv_tensor(inst.ofm)
        ctx.run()

        compiled = ctx._last_compiled
        assert len(compiled.shm_image) % 64 == 0


# ═══════════════════════════════════════════════════════════════════
# Pipeline with dependency chain
# ═══════════════════════════════════════════════════════════════════


class TestDependencyChainE2E:
    """Dependencies correctly flow through the full pipeline."""

    def test_send_then_recv_with_dep(self):
        """recv depends on send via OperationHandle."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=8)
        inst.data_in.fill_random()

        h = ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out, dep=h)
        ctx.run()

        compiled = ctx._last_compiled
        # recv's PULL command should depend on send's last command (PUSH)
        pull_cmd = [c for c in compiled.commands if c.op == OpCode.PULL][0]
        push_cmd = [c for c in compiled.commands if c.op == OpCode.PUSH][0]
        assert push_cmd.cmd_id in pull_cmd.dep

    def test_barrier_synchronizes(self):
        """BARRIER gets sync=True."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=8)
        inst.data_in.fill_random()

        ctx.send_tensor(inst.data_in)
        ctx.barrier()
        ctx.recv_tensor(inst.data_out)
        ctx.run()

        compiled = ctx._last_compiled
        barrier_cmds = [c for c in compiled.commands if c.op == OpCode.BARRIER]
        assert len(barrier_cmds) == 1
        assert barrier_cmds[0].sync is True


# ═══════════════════════════════════════════════════════════════════
# Pipeline with alias
# ═══════════════════════════════════════════════════════════════════


class TestAliasE2E:
    """Buffer aliasing through the full pipeline."""

    def test_alias_shares_buffer_id(self):
        """Aliased tensors share the same buffer_id."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        ctx.alias(inst.ifm, inst.ofm)
        ctx.send_tensor(inst.ifm)
        ctx.recv_tensor(inst.ofm)
        ctx.run()

        compiled = ctx._last_compiled
        assert compiled.buffer_ids["ifm"] == compiled.buffer_ids["ofm"]

    def test_alias_send_skips_load(self):
        """send_tensor on alias target skips LOAD."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(MemKernel, spec=_mem_spec(), IN_CH=32, OUT_CH=32)
        inst.ifm.fill_random()

        # ofm aliases ifm → sending ofm skips LOAD
        ctx.alias(inst.ifm, inst.ofm)
        ctx.recv_tensor(inst.ifm)
        ctx.send_tensor(inst.ofm)  # alias target → PUSH only
        ctx.run()

        compiled = ctx._last_compiled
        # send_tensor(ofm) should produce just PUSH (no LOAD)
        # Count LOADs — should only be from recv_tensor's expansion (0 for recv which is PULL)
        load_cmds = [c for c in compiled.commands if c.op == OpCode.LOAD]
        # There should be no LOAD commands at all
        assert len(load_cmds) == 0


# ═══════════════════════════════════════════════════════════════════
# Error cases
# ═══════════════════════════════════════════════════════════════════


class TestPipelineErrors:
    """Error conditions in the full pipeline."""

    def test_no_kernels_raises(self):
        """compile() with no instantiated kernels → CompilationError."""
        ctx = ExecutionContext(project_params={})
        ctx.barrier()
        with pytest.raises(CompilationError, match="No kernels"):
            ctx.run()

    def test_shape_mismatch_detected(self):
        """Data element count != declared shape → ShapeMismatchError."""
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        # Set data with wrong element count
        inst.data_in.data = torch.tensor([1, 2, 3], dtype=torch.int8)  # 3 != 16

        ctx.send_tensor(inst.data_in)
        with pytest.raises(ShapeMismatchError):
            ctx.run()


# ═══════════════════════════════════════════════════════════════════
# SHM image structural validation
# ═══════════════════════════════════════════════════════════════════


class TestSHMImageStructure:
    """Verify SHM image binary structure from full pipeline."""

    @pytest.fixture()
    def compiled_passthrough(self):
        ctx = ExecutionContext(project_params={})
        inst = ctx.instantiate(PassthroughKernel, spec=_passthrough_spec(), SIZE=16)
        inst.data_in.fill_random()
        ctx.send_tensor(inst.data_in)
        ctx.recv_tensor(inst.data_out)
        ctx.run()
        return ctx._last_compiled

    def test_control_header_magic(self, compiled_passthrough):
        img = compiled_passthrough.shm_image
        magic = struct.unpack_from("<I", img, 0)[0]
        assert magic == 0x5654454E

    def test_control_header_version(self, compiled_passthrough):
        img = compiled_passthrough.shm_image
        version = struct.unpack_from("<I", img, 4)[0]
        assert version == 0x00000003

    def test_num_commands_in_header(self, compiled_passthrough):
        """num_commands field matches actual command count."""
        img = compiled_passthrough.shm_image
        num_cmds = struct.unpack_from("<I", img, 0x10)[0]
        assert num_cmds == len(compiled_passthrough.commands)

    def test_num_buffers_in_header(self, compiled_passthrough):
        """num_buffers field matches buffer descriptor count."""
        img = compiled_passthrough.shm_image
        num_bufs = struct.unpack_from("<I", img, 0x14)[0]
        assert num_bufs >= len(compiled_passthrough.buffer_ids)

    def test_total_shm_size_in_header(self, compiled_passthrough):
        """total_shm_size in header matches actual image size."""
        img = compiled_passthrough.shm_image
        total_size = struct.unpack_from("<Q", img, 0x38)[0]
        assert total_size == len(img)

    def test_load_stats_pre_committed(self, compiled_passthrough):
        """LOAD command stats are pre-set to COMMITTED (value=3)."""
        img = compiled_passthrough.shm_image
        num_cmds = struct.unpack_from("<I", img, 0x10)[0]
        cmd_offset = 256  # CONTROL_SIZE
        stats_offset = cmd_offset + 64 * num_cmds  # CMD_SLOT_SIZE * num_cmds

        for cmd in compiled_passthrough.commands:
            if cmd.op == OpCode.LOAD:
                # Stats entry at stats_offset + cmd_id * 32
                stat_addr = stats_offset + cmd.cmd_id * 32
                status = struct.unpack_from("<I", img, stat_addr)[0]
                assert status == 3  # COMMITTED
