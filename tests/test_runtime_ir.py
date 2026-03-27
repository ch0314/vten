"""Phase 2 tests — Stage 6: IR Lowering.

Spec reference: 02_runtime_engine.md §12, 00_data_models.md §9
NPU 3D patterns: npu_3d_analysis.md §7

Tests Operation → Command expansion, role determination,
dependency resolution, register lowering, and configure().
"""

from __future__ import annotations

import struct

import pytest
import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.errors import (
    CompilationError,
    DependencyLimitError,
)
from vten.kernel.base import Kernel, RegisterHandle, register
from vten.kernel.tensor import Tensor
from vten.runtime.context import AliasRegistry
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


# ── Helpers ──────────────────────────────────────────────────────


class PassthroughKernel(Kernel):
    data_in = Tensor(shape=(16,), dtype=torch.int8, interface="axis_in")
    data_out = Tensor(shape=(16,), dtype=torch.int8, interface="axis_out")

    def generate_inputs(self, seed=None):
        self.data_in.fill_random()


class MemKernel(Kernel):
    """Kernel with AXI4 memory-mapped interface and AXI4-Lite control."""
    ifm = Tensor(shape=(32,), dtype=torch.int8, interface="ddr", direction=Direction.HOST_TO_DEV)
    ofm = Tensor(shape=(32,), dtype=torch.int8, interface="ddr", direction=Direction.DEV_TO_HOST)

    def generate_inputs(self, seed=None):
        self.ifm.fill_random()


def _make_stream_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        parameters={},
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


def _make_mem_spec(registers=None) -> KernelSpec:
    if registers is None:
        registers = [
            RegisterSpec(name="in_ch", offset=0x010, auto_bind=AutoBindSpec(param="${IN_CH}")),
            RegisterSpec(name="vsync", offset=0x050, fields={"trigger": "0:0"}),
            RegisterSpec(name="layer_done", offset=0x054, fields={"done": "0:0"}),
        ]
    return KernelSpec(
        kernel_name="memkernel",
        rtl_top="rtl/mem.sv",
        parameters={"IN_CH": "${IN_CH}"},
        memory_regions={
            "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000, alignment=4096),
        },
        interfaces={
            "ctrl": InterfaceSpec(
                name="ctrl",
                rtl_port="s_axilite_ctrl",
                protocol=Protocol.AXI4L,
                addr_width=16,
                registers=registers,
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


def _make_ir_setup(spec, kernel_class, runtime_params=None, alias_registry=None):
    """Create (view, IRLowering, kernel_instance) for testing."""
    from vten.runtime.flattener import (
        ExposedTensor,
        FlattenedKernelView,
        InterfaceMapping,
        KernelInstance,
    )
    from vten.runtime.ir import IRLowering

    params = runtime_params or {}
    inst = KernelInstance(
        name=kernel_class.__name__,
        spec=spec,
        kernel_class=kernel_class,
        runtime_params=params,
    )
    inst.initialize({})

    mappings = []
    for iface in spec.interfaces.values():
        mappings.append(
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface=iface.name,
                mapping_type=MappingType.EXTERNAL,
                top_interface=iface.name,
                bank_name=None,
                bank_offset=0,
            )
        )

    exposed = {}
    for tensor in inst.tensors():
        direction = Direction.HOST_TO_DEV
        exposed[tensor.name] = ExposedTensor(
            name=tensor.name,
            origin_path=f"_self.{tensor.name}",
            origin_tensor=tensor,
            top_interface=tensor.interface,
            direction=direction,
        )

    view = FlattenedKernelView(
        name=inst.name,
        top_spec=spec,
        sub_kernels={"_self": inst},
        interface_mappings=mappings,
        exposed_tensors=exposed,
        probe_points=[],
        connections=[],
    )

    # Set serialized_size for all exposed tensors (needed for commands)
    for name, exp in exposed.items():
        packing = None
        try:
            iface = spec.get_interface(exp.top_interface)
            packing = iface.packing
        except KeyError:
            pass
        if packing:
            import math
            num_beats = math.ceil(exp.origin_tensor._element_count / packing.elements_per_beat)
            exp._serialized_size = num_beats * (packing.bus_width // 8)

    lowering = IRLowering(view, alias_registry)
    return view, lowering, inst


def _lower_ops(view, lowering, ops):
    """Helper to lower operations and return (commands, buffer_ids)."""
    return lowering.lower(ops)


# ═══════════════════════════════════════════════════════════════════
# §12.0 — Role Determination
# ═══════════════════════════════════════════════════════════════════


class TestRoleDetermination:
    """Protocol × OpCode → BFM Role mapping."""

    @pytest.fixture()
    def determine_role(self):
        from vten.runtime.ir import _determine_role
        return _determine_role

    def test_axi4s_push_is_master(self, determine_role):
        assert determine_role(Protocol.AXI4S, OpCode.PUSH) == Role.MASTER

    def test_axi4s_pull_is_slave(self, determine_role):
        assert determine_role(Protocol.AXI4S, OpCode.PULL) == Role.SLAVE

    def test_axi4_push_is_slave(self, determine_role):
        """AXI4: DUT is always master, BFM is slave."""
        assert determine_role(Protocol.AXI4, OpCode.PUSH) == Role.SLAVE

    def test_axi4_pull_is_slave(self, determine_role):
        assert determine_role(Protocol.AXI4, OpCode.PULL) == Role.SLAVE

    def test_axi4l_always_master(self, determine_role):
        """AXI4-Lite: BFM always master (drives register r/w)."""
        assert determine_role(Protocol.AXI4L, OpCode.WRITE_REG) == Role.MASTER
        assert determine_role(Protocol.AXI4L, OpCode.READ_REG) == Role.MASTER
        assert determine_role(Protocol.AXI4L, OpCode.POLL_REG) == Role.MASTER


# ═══════════════════════════════════════════════════════════════════
# §12.10 — parse_bit_range utility
# ═══════════════════════════════════════════════════════════════════


class TestParseBitRange:
    """parse_bit_range("hi:lo") → (hi, lo)."""

    @pytest.fixture()
    def parse_bit_range(self):
        from vten.runtime.ir import parse_bit_range
        return parse_bit_range

    def test_single_bit(self, parse_bit_range):
        assert parse_bit_range("0:0") == (0, 0)
        assert parse_bit_range("1:1") == (1, 1)

    def test_full_32bit(self, parse_bit_range):
        assert parse_bit_range("31:0") == (31, 0)

    def test_upper_32bit(self, parse_bit_range):
        assert parse_bit_range("63:32") == (63, 32)

    def test_multi_bit_field(self, parse_bit_range):
        assert parse_bit_range("3:0") == (3, 0)

    def test_whitespace_tolerance(self, parse_bit_range):
        assert parse_bit_range(" 7 : 4 ") == (7, 4)

    def test_invalid_hi_less_than_lo(self, parse_bit_range):
        with pytest.raises(ValueError, match="hi.*lo"):
            parse_bit_range("0:1")

    def test_invalid_format(self, parse_bit_range):
        with pytest.raises(ValueError):
            parse_bit_range("31")


# ═══════════════════════════════════════════════════════════════════
# §9 — Operation → Command expansion counts
# ═══════════════════════════════════════════════════════════════════


class TestExpansionCounts:
    """Each OpKind expands to the correct number of Commands."""

    def test_load_tensor_expands_to_1(self):
        """LOAD_TENSOR → 1 LOAD command."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.LOAD

    def test_store_tensor_skipped_for_axi4s(self):
        """STORE_TENSOR → 0 commands for AXI4-Stream (data read from SHM directly)."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.STORE_TENSOR, tensor=inst.get_tensor("data_out"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 0

    def test_push_tensor_single_port(self):
        """PUSH_TENSOR (no split) → 1 PUSH command."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.PUSH

    def test_pull_tensor_single_port(self):
        """PULL_TENSOR (no split) → 1 PULL command."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.PULL_TENSOR, tensor=inst.get_tensor("data_out"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.PULL

    def test_barrier_single(self):
        """BARRIER → 1 BARRIER command."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.BARRIER)
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.BARRIER
        assert cmds[0].sync is True

    def test_send_tensor_no_alias(self):
        """SEND_TENSOR (no alias) → 2 commands: LOAD + PUSH."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.SEND_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        assert cmds[0].op == OpCode.LOAD
        assert cmds[1].op == OpCode.PUSH

    def test_recv_tensor_no_alias_axi4(self):
        """RECV_TENSOR on AXI4 (no alias) → 2 commands: PULL + STORE."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(kind=OpKind.RECV_TENSOR, tensor=inst.get_tensor("ofm"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        assert cmds[0].op == OpCode.PULL
        assert cmds[1].op == OpCode.STORE

    def test_recv_tensor_no_alias_axi4s(self):
        """RECV_TENSOR on AXI4-Stream (no alias) → 1 PULL (no STORE for stream)."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.RECV_TENSOR, tensor=inst.get_tensor("data_out"))
        cmds, _ = _lower_ops(view, lowering, [op])
        # AXI4-Stream: PULL only, no STORE
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.PULL


# ═══════════════════════════════════════════════════════════════════
# §12.8 — chunked recv_tensor lowering
# ═══════════════════════════════════════════════════════════════════


class TestChunkedRecvTensorLowering:
    """recv_tensor(tensor, chunks=N) generates per-chunk PULL commands."""

    def test_chunked_recv_stream_generates_n_pulls(self):
        """chunks=4 on AXI4-Stream → 4 separate PULL commands."""
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel,
        )
        ops = [
            Operation(
                kind=OpKind.RECV_TENSOR,
                tensor=inst.get_tensor("data_out"),
                chunk_index=i,
                chunk_total=4,
                chunks_spec=4,
            )
            for i in range(4)
        ]
        cmds, buf_ids = _lower_ops(view, lowering, ops)
        # AXI4-Stream: PULL only per chunk, no STORE
        assert len(cmds) == 4
        assert all(c.op == OpCode.PULL for c in cmds)
        # Each chunk has a unique buffer_id
        chunk_bids = [c.buffer_id for c in cmds]
        assert len(set(chunk_bids)) == 4

    def test_chunked_recv_mem_generates_pull_store_pairs(self):
        """chunks=2 on AXI4 → 2 × (PULL + STORE) = 4 commands."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32},
        )
        ops = [
            Operation(
                kind=OpKind.RECV_TENSOR,
                tensor=inst.get_tensor("ofm"),
                chunk_index=i,
                chunk_total=2,
                chunks_spec=2,
            )
            for i in range(2)
        ]
        cmds, buf_ids = _lower_ops(view, lowering, ops)
        assert len(cmds) == 4
        assert cmds[0].op == OpCode.PULL
        assert cmds[1].op == OpCode.STORE
        assert cmds[2].op == OpCode.PULL
        assert cmds[3].op == OpCode.STORE
        # Different buffer_ids for each chunk
        assert cmds[0].buffer_id != cmds[2].buffer_id

    def test_chunked_recv_sizes_are_equal_split(self):
        """Each chunk's PULL size = total_serialized_size / N."""
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel,
        )
        tensor = inst.get_tensor("data_out")
        total_size = view.exposed_tensors["data_out"]._serialized_size
        ops = [
            Operation(
                kind=OpKind.RECV_TENSOR,
                tensor=tensor,
                chunk_index=i,
                chunk_total=4,
                chunks_spec=4,
            )
            for i in range(4)
        ]
        cmds, _ = _lower_ops(view, lowering, ops)
        expected_per_chunk = total_size // 4
        for cmd in cmds:
            assert cmd.size == expected_per_chunk

    def test_chunked_recv_buffer_id_naming(self):
        """Buffer IDs follow {name}:chunk_{i} pattern."""
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel,
        )
        ops = [
            Operation(
                kind=OpKind.RECV_TENSOR,
                tensor=inst.get_tensor("data_out"),
                chunk_index=i,
                chunk_total=3,
                chunks_spec=3,
            )
            for i in range(3)
        ]
        _, buf_ids = _lower_ops(view, lowering, ops)
        for i in range(3):
            assert f"data_out:chunk_{i}" in buf_ids
        # Logical name maps to chunk_0
        assert buf_ids["data_out"] == buf_ids["data_out:chunk_0"]

    def test_chunked_recv_with_list_chunks_spec(self):
        """chunks=[10, 6] gives proportional sizes."""
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel,
        )
        total_size = view.exposed_tensors["data_out"]._serialized_size
        ops = [
            Operation(
                kind=OpKind.RECV_TENSOR,
                tensor=inst.get_tensor("data_out"),
                chunk_index=i,
                chunk_total=2,
                chunks_spec=[10, 6],
            )
            for i in range(2)
        ]
        cmds, _ = _lower_ops(view, lowering, ops)
        assert len(cmds) == 2
        assert cmds[0].size == total_size * 10 // 16
        assert cmds[1].size == total_size * 6 // 16


class TestChunkedRecvTensorContext:
    """ExecutionContext.recv_tensor(chunks=...) API tests."""

    def test_recv_tensor_chunks_returns_list(self):
        """chunks=N returns list of N OperationHandles."""
        from vten.runtime.context import ExecutionContext

        ctx = ExecutionContext()
        tensor = Tensor(shape=(16,), dtype=torch.int8, interface="axis_out")
        tensor.name = "data_out"
        handles = ctx.recv_tensor(tensor, chunks=4)
        assert isinstance(handles, list)
        assert len(handles) == 4
        for i, h in enumerate(handles):
            assert h.op.chunk_index == i
            assert h.op.chunk_total == 4
            assert h.op.chunks_spec == 4

    def test_recv_tensor_no_chunks_returns_single_handle(self):
        """Without chunks, returns single OperationHandle (backward compat)."""
        from vten.runtime.context import ExecutionContext

        ctx = ExecutionContext()
        tensor = Tensor(shape=(16,), dtype=torch.int8, interface="axis_out")
        tensor.name = "data_out"
        handle = ctx.recv_tensor(tensor)
        assert isinstance(handle, OperationHandle)
        assert handle.op.chunk_index is None

    def test_recv_tensor_list_chunks_returns_list(self):
        """chunks=[5, 11] returns list of 2 handles."""
        from vten.runtime.context import ExecutionContext

        ctx = ExecutionContext()
        tensor = Tensor(shape=(16,), dtype=torch.int8, interface="axis_out")
        tensor.name = "data_out"
        handles = ctx.recv_tensor(tensor, chunks=[5, 11])
        assert len(handles) == 2
        assert handles[0].op.chunks_spec == [5, 11]


# ═══════════════════════════════════════════════════════════════════
# §12.7 — write_register lowering
# ═══════════════════════════════════════════════════════════════════


class TestWriteRegisterLowering:
    """write_register field resolution and encoding."""

    def test_register_name_matching(self):
        """Key matches registers[].name → full register value."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"in_ch": 42},
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.WRITE_REG
        assert cmds[0].reg_offset == 0x010
        assert cmds[0].reg_value == 42

    def test_field_name_matching(self):
        """Key matches registers[].fields → value shifted to bit position."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"trigger": 1},
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.WRITE_REG
        # vsync register offset=0x050, field "trigger" at "0:0"
        assert cmds[0].reg_offset == 0x050
        # (1 << 0) & 0x1 = 1
        assert cmds[0].reg_value == 1

    def test_npu_3d_vsync_write(self):
        """NPU vsync register: field 'trigger' at bit 0:0."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"vsync": 1},
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].reg_offset == 0x050
        assert cmds[0].reg_value == 1

    def test_multiple_fields_multiple_commands(self):
        """{"reg_a": val_a, "reg_b": val_b} → 2 WRITE_REG commands."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"in_ch": 32, "vsync": 1},
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        assert all(c.op == OpCode.WRITE_REG for c in cmds)


# ═══════════════════════════════════════════════════════════════════
# §12.9 — poll_register lowering
# ═══════════════════════════════════════════════════════════════════


class TestPollRegisterLowering:
    """poll_register mask/expected computation from bit range."""

    def test_single_bit_field(self):
        """field 'done' at '0:0' → mask=0x1, expected=0x1."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="done",
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.POLL_REG
        assert cmds[0].reg_offset == 0x054
        assert cmds[0].reg_mask == 0x1
        assert cmds[0].reg_expected == 0x1

    def test_multi_bit_field(self):
        """field 'status' at '3:0' → mask=0xF, expected=0xF."""
        regs = [
            RegisterSpec(name="status_reg", offset=0x060, fields={"status": "3:0"}),
        ]
        spec = _make_mem_spec(registers=regs)
        view, lowering, inst = _make_ir_setup(
            spec, MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="status",
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert cmds[0].reg_mask == 0xF
        assert cmds[0].reg_expected == 0xF

    def test_upper_bit_field(self):
        """field 'busy' at '1:1' → mask=0x2, expected=0x2."""
        regs = [
            RegisterSpec(name="flags_reg", offset=0x070, fields={"busy": "1:1"}),
        ]
        spec = _make_mem_spec(registers=regs)
        view, lowering, inst = _make_ir_setup(
            spec, MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="busy",
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert cmds[0].reg_mask == 0x2
        assert cmds[0].reg_expected == 0x2

    def test_npu_3d_layer_done_poll(self):
        """NPU layer_done at offset 0x054, field 'done' at '0:0'."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="done",
        )
        cmds, _ = _lower_ops(view, lowering, [op])
        assert cmds[0].reg_offset == 0x054
        assert cmds[0].reg_mask == 1
        assert cmds[0].reg_expected == 1


# ═══════════════════════════════════════════════════════════════════
# §12.5 — configure() lowering
# ═══════════════════════════════════════════════════════════════════


class TestConfigureLowering:
    """configure(kernel) → N × WRITE_REG for auto_bind registers."""

    def test_auto_bind_param_resolution(self):
        """auto_bind with param: "${IN_CH}" → resolved integer value."""
        from vten.runtime.binder import resolve_auto_binds

        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        view._register_bindings = resolve_auto_binds(view)
        op = Operation(kind=OpKind.CONFIGURE, kernel=None)
        cmds, _ = _lower_ops(view, lowering, [op])
        # Only 'in_ch' has auto_bind
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.WRITE_REG
        assert cmds[0].reg_offset == 0x010
        assert cmds[0].reg_value == 32

    def test_auto_bind_address_bits(self):
        """auto_bind with address + bits: split 64-bit address."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="addr_lsb", offset=0x038,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="31:0"),
            ),
            RegisterSpec(
                name="addr_msb", offset=0x03C,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="63:32"),
            ),
        ]
        spec = _make_mem_spec(registers=regs)
        view, lowering, inst = _make_ir_setup(
            spec, MemKernel, runtime_params={"IN_CH": 32}
        )
        view.exposed_tensors["ifm"].set_address(0x0000_1000_DEAD_BEEF)
        view._register_bindings = resolve_auto_binds(view)

        op = Operation(kind=OpKind.CONFIGURE, kernel=None)
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        lsb_cmd = [c for c in cmds if c.reg_offset == 0x038][0]
        msb_cmd = [c for c in cmds if c.reg_offset == 0x03C][0]
        assert lsb_cmd.reg_value == 0xDEAD_BEEF
        assert msb_cmd.reg_value == 0x0000_1000

    def test_auto_bind_size_bytes(self):
        """auto_bind with value: 'size_bytes'."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="buf_size", offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="size_bytes"),
            ),
        ]
        spec = _make_mem_spec(registers=regs)
        view, lowering, inst = _make_ir_setup(
            spec, MemKernel, runtime_params={"IN_CH": 32}
        )
        view.exposed_tensors["ifm"]._serialized_size = 2048
        view._register_bindings = resolve_auto_binds(view)

        op = Operation(kind=OpKind.CONFIGURE, kernel=None)
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].reg_value == 2048

    def test_first_cmd_gets_dep_rest_empty(self):
        """Only first WRITE_REG in configure() expansion gets dep."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(name="reg_a", offset=0x010, auto_bind=AutoBindSpec(param="${IN_CH}")),
            RegisterSpec(name="reg_b", offset=0x014, auto_bind=AutoBindSpec(param="${IN_CH}")),
        ]
        spec = _make_mem_spec(registers=regs)
        view, lowering, inst = _make_ir_setup(
            spec, MemKernel, runtime_params={"IN_CH": 32}
        )
        view._register_bindings = resolve_auto_binds(view)

        # Create a preceding op to generate a dep
        push_op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.get_tensor("ifm"))
        push_handle = OperationHandle(op=push_op)
        configure_op = Operation(
            kind=OpKind.CONFIGURE, kernel=None, dep=[push_handle]
        )
        cmds, _ = _lower_ops(view, lowering, [push_op, configure_op])

        # First configure cmd (index 1) gets dep, second (index 2) gets empty
        configure_cmds = [c for c in cmds if c.reg_offset in (0x010, 0x014)]
        assert len(configure_cmds) == 2
        assert len(configure_cmds[0].dep) > 0
        assert configure_cmds[1].dep == []


# ═══════════════════════════════════════════════════════════════════
# §12.4 — Dependency resolution
# ═══════════════════════════════════════════════════════════════════


class TestDependencyResolution:
    """OperationHandle → cmd_id dependency mapping."""

    def test_dep_references_last_cmd(self):
        """Multi-command expansion: downstream dep → last cmd_id."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        # send_tensor → LOAD(id=0), PUSH(id=1)
        send_op = Operation(kind=OpKind.SEND_TENSOR, tensor=inst.get_tensor("data_in"))
        send_handle = OperationHandle(op=send_op)
        # barrier depends on send
        barrier_op = Operation(
            kind=OpKind.BARRIER,
            dep=[send_handle],
        )
        cmds, _ = _lower_ops(view, lowering, [send_op, barrier_op])
        # send → LOAD(0), PUSH(1); barrier → BARRIER(2) with dep=[1]
        barrier_cmd = cmds[2]
        assert barrier_cmd.op == OpCode.BARRIER
        assert 1 in barrier_cmd.dep

    def test_no_dep_empty_list(self):
        """Operation with dep=None → empty dep list."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert cmds[0].dep == []

    def test_commit_dep_on_last_command(self):
        """commit_dep applied to expansion's last command only."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        load_op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        load_handle = OperationHandle(op=load_op)
        # send_tensor expands to LOAD+PUSH; commit_dep on the PUSH
        send_op = Operation(
            kind=OpKind.SEND_TENSOR,
            tensor=inst.get_tensor("data_in"),
            commit_dep=[load_handle],
        )
        cmds, _ = _lower_ops(view, lowering, [load_op, send_op])
        # send → LOAD(1), PUSH(2); commit_dep on PUSH(2) → [0]
        push_cmd = [c for c in cmds if c.op == OpCode.PUSH][0]
        assert 0 in push_cmd.commit_dep


# ═══════════════════════════════════════════════════════════════════
# §16.2 V11 — Dependency limits
# ═══════════════════════════════════════════════════════════════════


class TestDependencyLimits:
    """Max 4 issue deps, max 4 commit deps per command."""

    def test_max_4_issue_deps(self):
        """≤4 issue deps is valid."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        # Create 4 independent loads
        ops = []
        handles = []
        for i in range(4):
            op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
            ops.append(op)
            handles.append(OperationHandle(op=op))
        # 5th op depends on all 4
        barrier = Operation(kind=OpKind.BARRIER, dep=handles)
        ops.append(barrier)
        cmds, _ = _lower_ops(view, lowering, ops)
        barrier_cmd = cmds[-1]
        assert len(barrier_cmd.dep) == 4

    def test_over_4_issue_deps_error(self):
        """>4 issue deps → DependencyLimitError."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        ops = []
        handles = []
        for i in range(5):
            op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
            ops.append(op)
            handles.append(OperationHandle(op=op))
        barrier = Operation(kind=OpKind.BARRIER, dep=handles)
        ops.append(barrier)
        with pytest.raises(DependencyLimitError, match="exceeds limit 4"):
            _lower_ops(view, lowering, ops)


# ═══════════════════════════════════════════════════════════════════
# §12.1 — Buffer ID allocation
# ═══════════════════════════════════════════════════════════════════


class TestBufferIDAllocation:
    """Exposed tensors get sequential buffer IDs."""

    def test_sequential_ids(self):
        """Buffer IDs start at 0, increment sequentially."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        _, buffer_ids = _lower_ops(view, lowering, [op])
        # Two exposed tensors: data_in and data_out
        assert len(buffer_ids) == 2
        ids = sorted(buffer_ids.values())
        assert ids == [0, 1]

    def test_alias_shares_buffer_id(self):
        """Alias target gets same buffer_id as source."""
        alias_reg = AliasRegistry()
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel, alias_registry=alias_reg
        )
        # Register alias: data_out → data_in (they share buffer)
        alias_reg.register(inst.get_tensor("data_in"), inst.get_tensor("data_out"))
        # Re-create lowering with alias registry
        from vten.runtime.ir import IRLowering
        lowering = IRLowering(view, alias_reg)
        op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        _, buffer_ids = _lower_ops(view, lowering, [op])
        assert buffer_ids["data_in"] == buffer_ids["data_out"]


# ═══════════════════════════════════════════════════════════════════
# §12.6 — Shorthand expansion (alias-aware)
# ═══════════════════════════════════════════════════════════════════


class TestShorthandExpansion:
    """send_tensor / recv_tensor alias-aware lowering."""

    def test_send_no_alias_load_push(self):
        """send_tensor (no alias) → LOAD + PUSH."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        op = Operation(kind=OpKind.SEND_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        assert cmds[0].op == OpCode.LOAD
        assert cmds[1].op == OpCode.PUSH

    def test_send_alias_push_only(self):
        """send_tensor (alias target) → PUSH only, LOAD skipped."""
        alias_reg = AliasRegistry()
        view, lowering, inst = _make_ir_setup(
            _make_stream_spec(), PassthroughKernel, alias_registry=alias_reg
        )
        # data_in is alias target of data_out (data already in buffer)
        alias_reg.register(inst.get_tensor("data_out"), inst.get_tensor("data_in"))
        from vten.runtime.ir import IRLowering
        lowering = IRLowering(view, alias_reg)

        op = Operation(kind=OpKind.SEND_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.PUSH

    def test_recv_no_alias_pull_store_axi4(self):
        """recv_tensor on AXI4 (no alias) → PULL + STORE."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(kind=OpKind.RECV_TENSOR, tensor=inst.get_tensor("ofm"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 2
        assert cmds[0].op == OpCode.PULL
        assert cmds[1].op == OpCode.STORE

    def test_recv_alias_pull_only(self):
        """recv_tensor (alias source) → PULL only, STORE skipped."""
        alias_reg = AliasRegistry()
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel,
            runtime_params={"IN_CH": 32},
            alias_registry=alias_reg,
        )
        # ofm is alias source (another tensor will reuse its buffer)
        alias_reg.register(inst.get_tensor("ofm"), inst.get_tensor("ifm"))
        from vten.runtime.ir import IRLowering
        lowering = IRLowering(view, alias_reg)

        op = Operation(kind=OpKind.RECV_TENSOR, tensor=inst.get_tensor("ofm"))
        cmds, _ = _lower_ops(view, lowering, [op])
        # AXI4 + alias source → PULL only
        assert len(cmds) == 1
        assert cmds[0].op == OpCode.PULL


# ═══════════════════════════════════════════════════════════════════
# §12.3 — PUSH/PULL with split interface
# ═══════════════════════════════════════════════════════════════════


class TestMultiPortPushPull:
    """Split interface generates N commands per port."""

    def test_multi_port_push_generates_n_commands(self):
        """Split interface with 4 ports → 4 PUSH commands."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        # Manually set port_buffers on exposed tensor
        exposed = view.exposed_tensors["data_in"]
        exposed._port_buffers = {
            f"port_{i}": bytes(64) for i in range(4)
        }
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        assert len(cmds) == 4
        assert all(c.op == OpCode.PUSH for c in cmds)

    def test_first_port_gets_dep(self):
        """Only first port command gets dep, rest empty."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        exposed = view.exposed_tensors["data_in"]
        exposed._port_buffers = {
            f"port_{i}": bytes(32) for i in range(3)
        }
        # Create a preceding load to generate a dep
        load_op = Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("data_in"))
        load_handle = OperationHandle(op=load_op)
        push_op = Operation(
            kind=OpKind.PUSH_TENSOR,
            tensor=inst.get_tensor("data_in"),
            dep=[load_handle],
        )
        cmds, _ = _lower_ops(view, lowering, [load_op, push_op])
        push_cmds = [c for c in cmds if c.op == OpCode.PUSH]
        assert len(push_cmds) == 3
        # First push has dep, rest don't
        assert len(push_cmds[0].dep) > 0
        assert push_cmds[1].dep == []
        assert push_cmds[2].dep == []

    def test_each_port_gets_unique_interface_id(self):
        """Each port command gets a distinct interface_id."""
        view, lowering, inst = _make_ir_setup(_make_stream_spec(), PassthroughKernel)
        exposed = view.exposed_tensors["data_in"]
        exposed._port_buffers = {
            "hbm_00": bytes(32),
            "hbm_01": bytes(32),
        }
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.get_tensor("data_in"))
        cmds, _ = _lower_ops(view, lowering, [op])
        iface_ids = [c.interface_id for c in cmds]
        assert len(set(iface_ids)) == 2  # each port gets unique ID


# ═══════════════════════════════════════════════════════════════════
# NPU 3D full IR lowering scenario
# ═══════════════════════════════════════════════════════════════════


class TestNPU3DFullIR:
    """End-to-end IR lowering for NPU 3D single invocation."""

    def test_expected_command_sequence(self):
        """NPU 3D simplified: load → configure → write_reg → poll → store."""
        from vten.runtime.binder import resolve_auto_binds

        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        view._register_bindings = resolve_auto_binds(view)

        ops = [
            Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("ifm")),
            Operation(kind=OpKind.CONFIGURE, kernel=None),
            Operation(
                kind=OpKind.WRITE_REGISTER,
                register_interface="ctrl",
                register_fields={"trigger": 1},
            ),
            Operation(
                kind=OpKind.POLL_REGISTER,
                register_interface="ctrl",
                register_field_name="done",
            ),
            Operation(kind=OpKind.STORE_TENSOR, tensor=inst.get_tensor("ofm")),
        ]
        cmds, buffer_ids = _lower_ops(view, lowering, ops)

        opcodes = [c.op for c in cmds]
        assert OpCode.LOAD in opcodes
        assert OpCode.WRITE_REG in opcodes
        assert OpCode.POLL_REG in opcodes
        assert OpCode.STORE in opcodes

    def test_cmd_ids_are_sequential(self):
        """All cmd_ids in output are 0, 1, 2, ..., N-1."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        ops = [
            Operation(kind=OpKind.LOAD_TENSOR, tensor=inst.get_tensor("ifm")),
            Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.get_tensor("ifm")),
            Operation(kind=OpKind.PULL_TENSOR, tensor=inst.get_tensor("ofm")),
            Operation(kind=OpKind.STORE_TENSOR, tensor=inst.get_tensor("ofm")),
        ]
        cmds, _ = _lower_ops(view, lowering, ops)
        cmd_ids = [c.cmd_id for c in cmds]
        assert cmd_ids == list(range(len(cmds)))

    def test_register_offsets_include_bank(self):
        """Register offsets use absolute_offset (with bank_offset)."""
        from vten.runtime.binder import resolve_auto_binds
        from vten.runtime.flattener import (
            ExposedTensor,
            FlattenedKernelView,
            InterfaceMapping,
            KernelInstance,
        )
        from vten.runtime.ir import IRLowering

        regs = [
            RegisterSpec(name="in_ch", offset=0x010, auto_bind=AutoBindSpec(param="${IN_CH}")),
        ]
        spec = _make_mem_spec(registers=regs)
        inst = KernelInstance(
            name="MemKernel",
            spec=spec,
            kernel_class=MemKernel,
            runtime_params={"IN_CH": 32},
        )
        inst.initialize({})

        # Use bank_offset = 0x2000
        mappings = [
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface="ctrl",
                mapping_type=MappingType.EXTERNAL_BANK,
                top_interface="ctrl",
                bank_name="sub_a",
                bank_offset=0x2000,
            ),
        ]
        exposed = {}
        for tensor in inst.tensors():
            exposed[tensor.name] = ExposedTensor(
                name=tensor.name,
                origin_path=f"_self.{tensor.name}",
                origin_tensor=tensor,
                top_interface="ddr",
                direction=Direction.HOST_TO_DEV,
            )

        view = FlattenedKernelView(
            name="test",
            top_spec=spec,
            sub_kernels={"_self": inst},
            interface_mappings=mappings,
            exposed_tensors=exposed,
            probe_points=[],
            connections=[],
        )
        view._register_bindings = resolve_auto_binds(view)

        lowering = IRLowering(view)
        op = Operation(kind=OpKind.CONFIGURE, kernel=None)
        cmds, _ = lowering.lower([op])

        # in_ch at offset 0x010 + bank 0x2000 = 0x2010
        assert len(cmds) == 1
        assert cmds[0].reg_offset == 0x2010


# ═══════════════════════════════════════════════════════════════════
# §12.7 — Register resolution errors (V13, V14)
# ═══════════════════════════════════════════════════════════════════


class TestRegisterErrors:
    """Register name/field not found → CompilationError."""

    def test_unknown_register_name(self):
        """write_register with non-existent register → CompilationError."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.WRITE_REGISTER,
            register_interface="ctrl",
            register_fields={"nonexistent_reg": 42},
        )
        with pytest.raises(CompilationError, match="not found"):
            _lower_ops(view, lowering, [op])

    def test_unknown_poll_field(self):
        """poll_register with non-existent field → CompilationError."""
        view, lowering, inst = _make_ir_setup(
            _make_mem_spec(), MemKernel, runtime_params={"IN_CH": 32}
        )
        op = Operation(
            kind=OpKind.POLL_REGISTER,
            register_interface="ctrl",
            register_field_name="nonexistent_field",
        )
        with pytest.raises(CompilationError, match="not found"):
            _lower_ops(view, lowering, [op])


# ═══════════════════════════════════════════════════════════════════
# Command dataclass field checks
# ═══════════════════════════════════════════════════════════════════


class TestCommandFields:
    """Verify Command dataclass has all required fields (§9)."""

    def test_command_import(self):
        from vten.spec.models import OpCode
        assert OpCode.LOAD.value == 1
        assert OpCode.COMPARE.value == 9

    def test_opcode_values(self):
        """OpCode enum values match SHM encoding."""
        assert OpCode.LOAD.value == 1
        assert OpCode.PUSH.value == 2
        assert OpCode.PULL.value == 3
        assert OpCode.STORE.value == 4
        assert OpCode.WRITE_REG.value == 5
        assert OpCode.READ_REG.value == 6
        assert OpCode.POLL_REG.value == 7
        assert OpCode.BARRIER.value == 8
        assert OpCode.COMPARE.value == 9

    def test_command_dataclass_fields(self):
        """Command dataclass has all fields from spec §9."""
        from vten.runtime.ir import Command
        cmd = Command(op=OpCode.LOAD, cmd_id=0)
        assert hasattr(cmd, "op")
        assert hasattr(cmd, "cmd_id")
        assert hasattr(cmd, "interface_id")
        assert hasattr(cmd, "buffer_id")
        assert hasattr(cmd, "protocol")
        assert hasattr(cmd, "phys_addr")
        assert hasattr(cmd, "size")
        assert hasattr(cmd, "role")
        assert hasattr(cmd, "dep")
        assert hasattr(cmd, "commit_dep")
        assert hasattr(cmd, "reg_offset")
        assert hasattr(cmd, "reg_value")
        assert hasattr(cmd, "reg_mask")
        assert hasattr(cmd, "reg_expected")
        assert hasattr(cmd, "probe")
        assert hasattr(cmd, "golden_buf")
        assert hasattr(cmd, "sync")
        assert hasattr(cmd, "port")

    def test_protocol_shm_encoding(self):
        """Protocol SHM values: AXI4S=1, AXI4=2, AXI4L=3."""
        assert Protocol.AXI4S.value == "axi4_stream"
        assert Protocol.AXI4.value == "axi4"
        assert Protocol.AXI4L.value == "axi4_lite"

    def test_role_shm_encoding(self):
        """Role SHM values: MASTER=0, SLAVE=1."""
        assert Role.MASTER.value == "master"
        assert Role.SLAVE.value == "slave"
