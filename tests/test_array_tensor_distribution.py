"""Tests for array interface tensor distribution (Phases B+C).

Validates:
- Tensor data block-split across array elements (Stage 3)
- Per-element buffer_id allocation (Stage 6)
- Per-element PUSH/PULL command generation (Stage 6)
- SHM packing with per-element buffers (Stage 7)
"""

from __future__ import annotations

import math

import pytest
import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.kernel.base import Kernel
from vten.kernel.tensor import Tensor
from vten.runtime.flattener import (
    ExposedTensor,
    FlattenedKernelView,
    InterfaceMapping,
    KernelInstance,
)
from vten.runtime.ir import BFMConfig, Command, IRLowering
from vten.spec.models import (
    ArraySpec,
    Direction,
    InterleaveSpec,
    InterfaceSpec,
    KernelSpec,
    MappingType,
    OpCode,
    OpKind,
    PackingScheme,
    Protocol,
    Role,
)


# ── Helpers ──────────────────────────────────────────────────────


class ArrayKernel(Kernel):
    """Kernel with tensor bound to an array interface."""
    wgt = Tensor(shape=(64,), dtype=torch.int8, interface="wgt")

    def generate_inputs(self, seed=None):
        self.wgt.fill_random()


class ArrayOutputKernel(Kernel):
    """Kernel with output tensor bound to an array interface."""
    result = Tensor(
        shape=(64,), dtype=torch.int8, interface="result_stream",
        direction=Direction.DEV_TO_HOST,
    )


def _make_array_spec(dimensions, flat_name_pattern=None) -> KernelSpec:
    """Create a KernelSpec with an array interface."""
    return KernelSpec(
        kernel_name="array_test",
        rtl_top="rtl/array_test.sv",
        interfaces={
            "wgt": InterfaceSpec(
                name="wgt",
                rtl_port="s_axis_wgt",
                protocol=Protocol.AXI4S,
                data_width=256,
                tensor="wgt",
                array=ArraySpec(
                    dimensions=dimensions,
                    flat_name_pattern=flat_name_pattern,
                ),
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


def _make_array_output_spec(dimensions) -> KernelSpec:
    """Create a KernelSpec with an array output interface."""
    return KernelSpec(
        kernel_name="array_out_test",
        rtl_top="rtl/array_out.sv",
        interfaces={
            "result_stream": InterfaceSpec(
                name="result_stream",
                rtl_port="m_axis_result",
                protocol=Protocol.AXI4S,
                data_width=256,
                tensor="result",
                array=ArraySpec(dimensions=dimensions),
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


def _make_array_ir_setup(spec, kernel_class, tensor_name="wgt"):
    """Create (view, IRLowering, inst) for array interface tests."""
    inst = KernelInstance(
        name=kernel_class.__name__,
        spec=spec,
        kernel_class=kernel_class,
        runtime_params={},
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
        direction = tensor.direction or Direction.HOST_TO_DEV
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

    # Simulate Stage 3: serialize + array split
    for name, exp in exposed.items():
        try:
            iface = spec.get_interface(exp.top_interface)
        except KeyError:
            continue
        packing = iface.packing
        if packing is None:
            continue
        if exp.direction == Direction.HOST_TO_DEV:
            if exp.origin_tensor.data is not None:
                from vten.runtime.serializer import StreamSerializer
                serializer = StreamSerializer(packing)
                exp._serialized = serializer.serialize(exp.origin_tensor.data)
                exp._serialized_size = len(exp._serialized)
            else:
                num_beats = math.ceil(
                    exp.origin_tensor._element_count / packing.elements_per_beat
                )
                exp._serialized_size = num_beats * (packing.bus_width // 8)
        else:
            num_beats = math.ceil(
                exp.origin_tensor._element_count / packing.elements_per_beat
            )
            exp._serialized = None
            exp._serialized_size = num_beats * (packing.bus_width // 8)

        # Array split (same logic as engine._serialize_tensors)
        if iface.array and not exp._port_buffers:
            flat_names = iface.array.flat_names(exp.top_interface)
            n = len(flat_names)
            if iface.array.interleave and exp._serialized is not None:
                from vten.runtime.serializer import MultiPortSerializer
                from vten.spec.models import InterleaveSpec, PortDef, SplitSpec
                pseudo_spec = SplitSpec(
                    mode="channel_interleave",
                    ports=[PortDef(name=fn, base_addr=0) for fn in flat_names],
                    interleave=iface.array.interleave,
                )
                splitter = MultiPortSerializer()
                exp._port_buffers = splitter.split_tensor(
                    exp._serialized, pseudo_spec
                )
                exp._port_mode = "channel_interleave"
                exp._interleave_unit = iface.array.interleave.unit
            elif iface.array.interleave and exp._serialized is None:
                per_elem_size = exp._serialized_size // n
                exp._port_buffers = {
                    fname: bytes(per_elem_size) for fname in flat_names
                }
                exp._port_mode = "channel_interleave"
                exp._interleave_unit = iface.array.interleave.unit
            elif exp._serialized is not None:
                data = exp._serialized
                chunk_size = len(data) // n
                remainder = len(data) % n
                exp._port_buffers = {}
                offset = 0
                for i, fname in enumerate(flat_names):
                    sz = chunk_size + (1 if i < remainder else 0)
                    exp._port_buffers[fname] = data[offset: offset + sz]
                    offset += sz
            else:
                per_elem_size = exp._serialized_size // n
                exp._port_buffers = {
                    fname: bytes(per_elem_size) for fname in flat_names
                }

    lowering = IRLowering(view)
    return view, lowering, inst


# ═══════════════════════════════════════════════════════════════════
# §1 — Array Element Buffer Splitting (Phase B)
# ═══════════════════════════════════════════════════════════════════


class TestArrayElementBufferSplit:
    """Stage 3: tensor data split across array elements."""

    def test_1d_array_split_count(self):
        """1D array [4] → 4 element buffers."""
        data = bytes(range(64))  # 64 bytes
        flat_names = ArraySpec(dimensions=[4]).flat_names("wgt")
        assert len(flat_names) == 4
        n = len(flat_names)
        chunk_size = len(data) // n
        buffers = {}
        offset = 0
        for fname in flat_names:
            buffers[fname] = data[offset: offset + chunk_size]
            offset += chunk_size
        assert len(buffers) == 4
        assert all(len(v) == 16 for v in buffers.values())

    def test_1d_array_split_data_integrity(self):
        """Split data reconstructs to original when concatenated."""
        data = bytes(range(128))
        flat_names = ArraySpec(dimensions=[4]).flat_names("wgt")
        n = len(flat_names)
        chunk_size = len(data) // n
        buffers = {}
        offset = 0
        for fname in flat_names:
            buffers[fname] = data[offset: offset + chunk_size]
            offset += chunk_size
        reconstructed = b"".join(buffers.values())
        assert reconstructed == data

    def test_2d_array_split(self):
        """2D array [2][3] → 6 element buffers."""
        data = bytes(range(60))  # 60 bytes, divisible by 6
        flat_names = ArraySpec(dimensions=[2, 3]).flat_names("wgt")
        assert len(flat_names) == 6
        assert flat_names == ["wgt_0_0", "wgt_0_1", "wgt_0_2",
                              "wgt_1_0", "wgt_1_1", "wgt_1_2"]
        n = len(flat_names)
        chunk_size = len(data) // n
        buffers = {}
        offset = 0
        for fname in flat_names:
            buffers[fname] = data[offset: offset + chunk_size]
            offset += chunk_size
        assert all(len(v) == 10 for v in buffers.values())

    def test_uneven_split_distributes_remainder(self):
        """When data size is not evenly divisible, remainder goes to first elements."""
        data = bytes(range(10))  # 10 bytes / 3 elements → 3+3+4? No: 4+3+3
        flat_names = ArraySpec(dimensions=[3]).flat_names("wgt")
        n = len(flat_names)
        chunk_size = len(data) // n
        remainder = len(data) % n
        buffers = {}
        offset = 0
        for i, fname in enumerate(flat_names):
            sz = chunk_size + (1 if i < remainder else 0)
            buffers[fname] = data[offset: offset + sz]
            offset += sz
        sizes = [len(v) for v in buffers.values()]
        assert sizes == [4, 3, 3]  # First element gets extra byte
        assert b"".join(buffers.values()) == data


# ═══════════════════════════════════════════════════════════════════
# §2 — Buffer ID Allocation (Phase C1)
# ═══════════════════════════════════════════════════════════════════


class TestArrayBufferIdAllocation:
    """IR buffer_id allocation for array tensors."""

    def test_per_element_buffer_ids(self):
        """Array tensor gets N distinct buffer_ids."""
        spec = _make_array_spec([4])
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)
        # Manually set up array element buffers
        exp = view.exposed_tensors["wgt"]
        data = bytes(64)
        exp._serialized = data
        exp._serialized_size = 64
        exp._port_buffers = {
            f"wgt_{i}": data[i*16:(i+1)*16] for i in range(4)
        }
        # Re-create lowering to pick up array buffers
        lowering = IRLowering(view)
        buffer_ids = lowering._allocate_buffer_ids()
        # Should have 4 element keys + 1 logical key
        element_keys = [k for k in buffer_ids if ":" in k]
        assert len(element_keys) == 4
        # All element buffer_ids should be distinct
        element_bids = [buffer_ids[k] for k in element_keys]
        assert len(set(element_bids)) == 4
        # Logical name maps to first element
        assert buffer_ids["wgt"] == buffer_ids["wgt:wgt_0"]

    def test_buffer_ids_sequential(self):
        """Element buffer_ids are sequential."""
        spec = _make_array_spec([3])
        view, _, inst = _make_array_ir_setup(spec, ArrayKernel)
        exp = view.exposed_tensors["wgt"]
        exp._port_buffers = {
            f"wgt_{i}": bytes(10) for i in range(3)
        }
        lowering = IRLowering(view)
        buffer_ids = lowering._allocate_buffer_ids()
        bids = [buffer_ids[f"wgt:wgt_{i}"] for i in range(3)]
        assert bids == [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════
# §3 — Per-Element PUSH/PULL Commands (Phase C2)
# ═══════════════════════════════════════════════════════════════════


class TestArrayPushPullCommands:
    """IR lowering generates per-element commands for array interfaces."""

    def _setup_with_data(self, dimensions):
        """Create setup with real serialized data for array interface."""
        spec = _make_array_spec(dimensions)
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)
        exp = view.exposed_tensors["wgt"]
        # Create mock serialized data
        total_elements = 1
        for d in dimensions:
            total_elements *= d
        data = bytes(range(256)) * (total_elements * 8 // 256 + 1)
        data = data[:total_elements * 8]  # 8 bytes per element
        exp._serialized = data
        exp._serialized_size = len(data)

        flat_names = spec.interfaces["wgt"].array.flat_names("wgt")
        n = len(flat_names)
        chunk_size = len(data) // n
        exp._port_buffers = {}
        offset = 0
        for fname in flat_names:
            exp._port_buffers[fname] = data[offset: offset + chunk_size]
            offset += chunk_size

        lowering = IRLowering(view)
        return view, lowering, inst, exp

    def test_push_generates_n_commands(self):
        """push_tensor on [4] array → 4 PUSH commands."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, buffer_ids = lowering.lower(ops)
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        assert len(push_cmds) == 4

    def test_push_different_interface_ids(self):
        """Each PUSH targets a different interface_id."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        iface_ids = [c.interface_id for c in push_cmds]
        assert len(set(iface_ids)) == 4

    def test_push_different_buffer_ids(self):
        """Each PUSH references a different buffer_id."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        bids = [c.buffer_id for c in push_cmds]
        assert len(set(bids)) == 4

    def test_push_sizes_sum_to_total(self):
        """Sum of per-element sizes equals total serialized size."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        total = sum(c.size for c in push_cmds)
        assert total == exp._serialized_size

    def test_push_each_cmd_depends_on_its_load(self):
        """Each PUSH depends on its corresponding LOAD."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        load_cmds = [c for c in commands if c.op == OpCode.LOAD]
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        # Each PUSH should depend on its corresponding LOAD
        for i, push_cmd in enumerate(push_cmds):
            assert push_cmd.dep == [load_cmds[i].cmd_id]

    def test_pull_generates_n_commands(self):
        """pull_tensor on [3] array → 3 PULL commands."""
        spec = _make_array_output_spec([3])
        view, lowering, inst = _make_array_ir_setup(
            spec, ArrayOutputKernel, tensor_name="result"
        )
        exp = view.exposed_tensors["result"]
        # Verify array element buffers were created
        assert exp._port_buffers is not None
        assert len(exp._port_buffers) == 3

        ops = [
            Operation(kind=OpKind.PULL_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        pull_cmds = [c for c in commands if c.op == OpCode.PULL]
        assert len(pull_cmds) == 3

    def test_2d_array_push(self):
        """2D array [2][2] → 4 PUSH commands with correct flat names."""
        view, lowering, inst, exp = self._setup_with_data([2, 2])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        assert len(push_cmds) == 4

    def test_cmd_ids_sequential(self):
        """All generated commands have sequential cmd_ids."""
        view, lowering, inst, exp = self._setup_with_data([4])
        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        cmd_ids = [c.cmd_id for c in commands]
        assert cmd_ids == list(range(len(commands)))


# ═══════════════════════════════════════════════════════════════════
# §4 — Push/Pull Tensor with Array (Phase C2)
# ═══════════════════════════════════════════════════════════════════


class TestArrayPushPullExpandedCommands:
    """push_tensor/pull_tensor expand to per-element LOAD+PUSH / PULL commands."""

    def test_push_tensor_generates_load_push_pairs(self):
        """push_tensor on [2] array → 2 LOAD + 2 PUSH = 4 commands."""
        spec = _make_array_spec([2])
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)
        exp = view.exposed_tensors["wgt"]
        # Setup with data
        data = bytes(64)
        exp._serialized = data
        exp._serialized_size = 64
        flat_names = spec.interfaces["wgt"].array.flat_names("wgt")
        exp._port_buffers = {
            fn: data[i*32:(i+1)*32] for i, fn in enumerate(flat_names)
        }
        lowering = IRLowering(view)

        ops = [
            Operation(kind=OpKind.PUSH_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        load_cmds = [c for c in commands if c.op == OpCode.LOAD]
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        assert len(load_cmds) == 2
        assert len(push_cmds) == 2
        # Each PUSH depends on its LOAD
        for i, push_cmd in enumerate(push_cmds):
            assert push_cmd.dep == [load_cmds[i].cmd_id]

    def test_pull_tensor_generates_pull_commands(self):
        """pull_tensor on [2] array for AXI4-Stream → 2 PULL (no STORE)."""
        spec = _make_array_output_spec([2])
        view, lowering, inst = _make_array_ir_setup(
            spec, ArrayOutputKernel, tensor_name="result"
        )
        exp = view.exposed_tensors["result"]
        assert exp._port_buffers is not None

        ops = [
            Operation(kind=OpKind.PULL_TENSOR, tensor=exp.origin_tensor),
        ]
        commands, _ = lowering.lower(ops)
        pull_cmds = [c for c in commands if c.op == OpCode.PULL]
        store_cmds = [c for c in commands if c.op == OpCode.STORE]
        # AXI4-Stream: no STORE needed
        assert len(pull_cmds) == 2
        assert len(store_cmds) == 0


# ═══════════════════════════════════════════════════════════════════
# §5 — Full Pipeline Integration (Stage 3→6→7)
# ═══════════════════════════════════════════════════════════════════


class TestArrayFullPipeline:
    """End-to-end: RuntimeEngine compile with array interface."""

    def test_compile_with_array_interface(self):
        """Full compile pipeline with array interface produces valid SHM."""
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([4])
        kernel_class = ArrayKernel

        # Create kernel instance via context-like setup
        inst = KernelInstance(
            name="ArrayKernel",
            spec=spec,
            kernel_class=kernel_class,
            runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        # Record push operation
        ops = [
            Operation(
                kind=OpKind.PUSH_TENSOR,
                tensor=inst.kernel_class_instance.wgt,
            ),
        ]

        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst},
            ops=ops,
            project_params={},
        )
        result = engine.compile(target="sim")

        # Verify commands: should have 4 PUSH commands
        push_cmds = [c for c in result.commands if c.op == OpCode.PUSH]
        assert len(push_cmds) == 4

        # Verify distinct interface_ids
        iface_ids = {c.interface_id for c in push_cmds}
        assert len(iface_ids) == 4

        # Verify distinct buffer_ids
        bids = {c.buffer_id for c in push_cmds}
        assert len(bids) == 4

        # Verify SHM image is non-empty
        assert len(result.shm_image) > 0

        # Verify tensor_data has per-element entries
        assert len(result.tensor_data) == 4

    def test_compile_push_tensor_with_array(self):
        """push_tensor with array: LOAD+PUSH pairs in SHM."""
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([2])
        inst = KernelInstance(
            name="ArrayKernel",
            spec=spec,
            kernel_class=ArrayKernel,
            runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [
            Operation(
                kind=OpKind.PUSH_TENSOR,
                tensor=inst.kernel_class_instance.wgt,
            ),
        ]

        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst},
            ops=ops,
            project_params={},
        )
        result = engine.compile(target="sim")

        load_cmds = [c for c in result.commands if c.op == OpCode.LOAD]
        push_cmds = [c for c in result.commands if c.op == OpCode.PUSH]
        assert len(load_cmds) == 2
        assert len(push_cmds) == 2

    def test_bfm_configs_expanded_for_array(self):
        """BFM configs have one entry per array element."""
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([4])
        inst = KernelInstance(
            name="ArrayKernel",
            spec=spec,
            kernel_class=ArrayKernel,
            runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [
            Operation(
                kind=OpKind.PUSH_TENSOR,
                tensor=inst.kernel_class_instance.wgt,
            ),
        ]

        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst},
            ops=ops,
            project_params={},
        )
        result = engine.compile(target="sim")

        # 4 BFMs for the 4-element array
        assert len(result.bfm_configs) == 4
        names = {bfm.interface_name for bfm in result.bfm_configs}
        assert names == {"wgt_0", "wgt_1", "wgt_2", "wgt_3"}


# ═══════════════════════════════════════════════════════════════════
# §6 — Codegen Integration (Phase D+E)
# ═══════════════════════════════════════════════════════════════════


class TestArrayCodegenIntegration:
    """Compile → Codegen: verify tb_top.sv renders correctly for arrays."""

    def test_tb_top_wire_declarations(self, tmp_path):
        """tb_top.sv has wire declarations for each array element."""
        from vten.codegen.sv_generator import SVGenerator
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([4])
        inst = KernelInstance(
            name="ArrayKernel", spec=spec,
            kernel_class=ArrayKernel, runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [Operation(
            kind=OpKind.PUSH_TENSOR,
            tensor=inst.kernel_class_instance.wgt,
        )]
        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst}, ops=ops, project_params={},
        )
        result = engine.compile(target="sim")

        gen = SVGenerator(spec, result.bfm_configs, {})
        files = gen.generate(str(tmp_path), num_commands=len(result.commands))
        assert "tb_top.sv" in files

        content = (tmp_path / "tb_top.sv").read_text()
        # Verify 4 element wire declarations
        for i in range(4):
            assert f"s_axis_wgt_{i}_tdata" in content
            assert f"s_axis_wgt_{i}_tvalid" in content
        # Verify 4 BFM instantiations
        for i in range(4):
            assert f"bfm_wgt_{i}" in content

    def test_tb_top_iface_to_bfm_mapping(self, tmp_path):
        """iface_to_bfm mapping covers all array element interface IDs."""
        from vten.codegen.sv_generator import SVGenerator
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([3])
        inst = KernelInstance(
            name="ArrayKernel", spec=spec,
            kernel_class=ArrayKernel, runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [Operation(
            kind=OpKind.PUSH_TENSOR,
            tensor=inst.kernel_class_instance.wgt,
        )]
        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst}, ops=ops, project_params={},
        )
        result = engine.compile(target="sim")

        gen = SVGenerator(spec, result.bfm_configs, {})
        files = gen.generate(str(tmp_path), num_commands=len(result.commands))

        content = (tmp_path / "tb_top.sv").read_text()
        # Each array element's interface_id maps to a BFM
        for i in range(3):
            assert f"iface_to_bfm[{i}] = {i}" in content

    def test_2d_array_codegen(self, tmp_path):
        """2D array [2][2] generates 4 BFMs with correct flat names."""
        from vten.codegen.sv_generator import SVGenerator
        from vten.runtime.engine import RuntimeEngine

        spec = _make_array_spec([2, 2], flat_name_pattern="wgt_{i}_{j}")
        inst = KernelInstance(
            name="ArrayKernel", spec=spec,
            kernel_class=ArrayKernel, runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [Operation(
            kind=OpKind.PUSH_TENSOR,
            tensor=inst.kernel_class_instance.wgt,
        )]
        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst}, ops=ops, project_params={},
        )
        result = engine.compile(target="sim")

        assert len(result.bfm_configs) == 4
        gen = SVGenerator(spec, result.bfm_configs, {})
        files = gen.generate(str(tmp_path), num_commands=len(result.commands))

        content = (tmp_path / "tb_top.sv").read_text()
        for i in range(2):
            for j in range(2):
                assert f"s_axis_wgt_{i}_{j}_tdata" in content
                assert f"bfm_wgt_{i}_{j}" in content

    def test_shm_data_integrity(self):
        """SHM image contains correct split data for each element buffer."""
        import struct
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.shm import CONTROL_SIZE, CMD_SLOT_SIZE, STATS_SLOT_SIZE, BUF_DESC_SIZE, CACHE_LINE

        spec = _make_array_spec([2])
        inst = KernelInstance(
            name="ArrayKernel", spec=spec,
            kernel_class=ArrayKernel, runtime_params={},
        )
        inst.initialize({})
        inst.kernel_class_instance.wgt.fill_random()

        ops = [Operation(
            kind=OpKind.PUSH_TENSOR,
            tensor=inst.kernel_class_instance.wgt,
        )]
        engine = RuntimeEngine(
            kernels={"ArrayKernel": inst}, ops=ops, project_params={},
        )
        result = engine.compile(target="sim")

        # Verify we have 2 distinct buffer_ids in tensor_data
        assert len(result.tensor_data) == 2

        # Concatenated element data should equal full serialized tensor
        bid_keys = sorted(result.tensor_data.keys())
        reconstructed = b"".join(result.tensor_data[k] for k in bid_keys)

        # The full serialized size should match
        view = result.flattened_view
        exp = view.exposed_tensors["wgt"]
        assert len(reconstructed) == exp._serialized_size


# ═══════════════════════════════════════════════════════════════════
# §7 — Array Interleave Mode
# ═══════════════════════════════════════════════════════════════════


def _make_array_interleave_spec(
    dimensions, unit, flat_name_pattern=None,
) -> KernelSpec:
    """Create a KernelSpec with an interleaved array interface."""
    return KernelSpec(
        kernel_name="array_interleave_test",
        rtl_top="rtl/array_interleave.sv",
        interfaces={
            "wgt": InterfaceSpec(
                name="wgt",
                rtl_port="s_axis_wgt",
                protocol=Protocol.AXI4S,
                data_width=256,
                tensor="wgt",
                array=ArraySpec(
                    dimensions=dimensions,
                    flat_name_pattern=flat_name_pattern,
                    interleave=InterleaveSpec(unit=unit),
                ),
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


def _make_array_interleave_output_spec(dimensions, unit) -> KernelSpec:
    """Create a KernelSpec with an interleaved array output interface."""
    return KernelSpec(
        kernel_name="array_interleave_out",
        rtl_top="rtl/array_interleave_out.sv",
        interfaces={
            "result_stream": InterfaceSpec(
                name="result_stream",
                rtl_port="m_axis_result",
                protocol=Protocol.AXI4S,
                data_width=256,
                tensor="result",
                array=ArraySpec(
                    dimensions=dimensions,
                    interleave=InterleaveSpec(unit=unit),
                ),
                packing=PackingScheme(element_width=8, elements_per_beat=32),
            ),
        },
    )


class TestArrayInterleaveDistribution:
    """Array with interleave: round-robin beat distribution across ports."""

    def test_interleave_3_ports_round_robin(self):
        """3-port interleave (unit=32): beats distributed round-robin."""
        # 6 beats × 32 bytes = 192 bytes → 2 beats per port
        data = bytes(range(192))
        unit = 32
        n_ports = 3
        flat_names = ArraySpec(dimensions=[n_ports]).flat_names("psum")

        from vten.runtime.serializer import MultiPortSerializer
        from vten.spec.models import PortDef, SplitSpec

        spec = SplitSpec(
            mode="channel_interleave",
            ports=[PortDef(name=n, base_addr=0) for n in flat_names],
            interleave=InterleaveSpec(unit=unit),
        )
        result = MultiPortSerializer().split_tensor(data, spec)

        assert len(result) == 3
        # port 0 gets beats 0, 3 (bytes 0-31, 96-127)
        assert result["psum_0"][:32] == data[0:32]
        assert result["psum_0"][32:64] == data[96:128]
        # port 1 gets beats 1, 4 (bytes 32-63, 128-159)
        assert result["psum_1"][:32] == data[32:64]
        assert result["psum_1"][32:64] == data[128:160]
        # port 2 gets beats 2, 5 (bytes 64-95, 160-191)
        assert result["psum_2"][:32] == data[64:96]
        assert result["psum_2"][32:64] == data[160:192]

    def test_interleave_6_ports_npu_pattern(self):
        """NPU MAC pattern: 6 streams, beat-interleaved."""
        unit = 32  # 256-bit bus = 32 bytes
        n_ports = 6
        total_beats = 12  # 2 beats per port
        data = bytes(i % 256 for i in range(total_beats * unit))

        flat_names = ArraySpec(dimensions=[n_ports]).flat_names("psum")
        from vten.runtime.serializer import MultiPortSerializer
        from vten.spec.models import PortDef, SplitSpec

        spec = SplitSpec(
            mode="channel_interleave",
            ports=[PortDef(name=n, base_addr=0) for n in flat_names],
            interleave=InterleaveSpec(unit=unit),
        )
        result = MultiPortSerializer().split_tensor(data, spec)

        assert len(result) == 6
        for i, fn in enumerate(flat_names):
            assert len(result[fn]) == 2 * unit  # 2 beats each
            # First beat of port i = beat i of original
            assert result[fn][:unit] == data[i * unit : (i + 1) * unit]
            # Second beat of port i = beat (6+i) of original
            assert result[fn][unit : 2 * unit] == data[(6 + i) * unit : (7 + i) * unit]

    def test_interleave_reassemble_roundtrip(self):
        """Interleave → reassemble recovers original data."""
        from vten.runtime.serializer import MultiPortSerializer
        from vten.spec.models import PortDef, SplitSpec

        unit = 32
        n_ports = 4
        data = bytes(i % 256 for i in range(n_ports * 8 * unit))  # 8 beats/port

        spec = SplitSpec(
            mode="channel_interleave",
            ports=[PortDef(name=f"p{i}", base_addr=0) for i in range(n_ports)],
            interleave=InterleaveSpec(unit=unit),
        )
        split = MultiPortSerializer().split_tensor(data, spec)
        reassembled = MultiPortSerializer.reassemble(split, unit)
        assert reassembled == data

    def test_interleave_via_ir_setup(self):
        """Full IR setup with interleaved array generates correct _port_buffers."""
        spec = _make_array_interleave_spec([3], unit=32)
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)

        exp = view.exposed_tensors["wgt"]
        assert exp._port_mode == "channel_interleave"
        assert exp._interleave_unit == 32
        assert len(exp._port_buffers) == 3
        assert set(exp._port_buffers.keys()) == {"wgt_0", "wgt_1", "wgt_2"}

    def test_interleave_output_empty_buffers(self):
        """DEV_TO_HOST array interleave allocates empty per-port buffers."""
        spec = _make_array_interleave_output_spec([4], unit=32)
        view, lowering, inst = _make_array_ir_setup(
            spec, ArrayOutputKernel, tensor_name="result",
        )

        exp = view.exposed_tensors["result"]
        assert exp._port_mode == "channel_interleave"
        assert exp._interleave_unit == 32
        assert len(exp._port_buffers) == 4
        # All buffers should be empty (zeros)
        for port_data in exp._port_buffers.values():
            assert port_data == bytes(len(port_data))

    def test_interleave_per_port_buffer_ids(self):
        """Interleaved array gets distinct buffer_id per port."""
        spec = _make_array_interleave_spec([3], unit=32)
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)

        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.kernel_class_instance.wgt)
        lowering._buffer_ids = lowering._allocate_buffer_ids([op])

        exp = view.exposed_tensors["wgt"]
        bids = set()
        for port_name in exp._port_buffers:
            key = f"{exp.name}:{port_name}"
            bid = lowering._buffer_ids.get(key)
            assert bid is not None, f"Missing buffer_id for {key}"
            bids.add(bid)
        assert len(bids) == 3

    def test_interleave_push_commands(self):
        """Interleaved array generates N PUSH commands."""
        spec = _make_array_interleave_spec([3], unit=32)
        view, lowering, inst = _make_array_ir_setup(spec, ArrayKernel)

        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=inst.kernel_class_instance.wgt)
        commands, _ = lowering.lower([op])
        push_cmds = [c for c in commands if c.op == OpCode.PUSH]
        assert len(push_cmds) == 3
        iface_ids = {c.interface_id for c in push_cmds}
        buf_ids = {c.buffer_id for c in push_cmds}
        assert len(iface_ids) == 3
        assert len(buf_ids) == 3
