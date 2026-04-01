"""Phase 2 tests — Stage 0: Composite Flattening.

Spec reference: 02_runtime_engine.md §5, 00_data_models.md §7
NPU 3D patterns: npu_3d_analysis.md §4

Tests KernelInstance, FlattenedKernelView, ExposedTensor,
InterfaceMapping, and direction inference.
"""

from __future__ import annotations

import copy

import pytest
import torch

from vten.errors import BindingError, ValidationError
from vten.kernel.base import Kernel, RegisterHandle, register
from vten.kernel.tensor import Tensor
from vten.spec.models import (
    AutoBindSpec,
    Direction,
    InterfaceSpec,
    KernelSpec,
    MappingType,
    MemoryRegion,
    PackingScheme,
    Protocol,
    RegisterBankSpec,
    RegisterSpec,
    Role,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_passthrough_spec() -> KernelSpec:
    """Minimal AXI4-Stream passthrough kernel spec."""
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        parameters={"SIZE": "${SIZE}"},
        interfaces={
            "axi_stream_in": InterfaceSpec(
                name="axi_stream_in",
                rtl_port="s_axis_in",
                protocol=Protocol.AXI4S,
                tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axi_stream_out": InterfaceSpec(
                name="axi_stream_out",
                rtl_port="m_axis_out",
                protocol=Protocol.AXI4S,
                tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


class PassthroughKernel(Kernel):
    data_in = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axi_stream_in")
    data_out = Tensor(shape=("${SIZE}",), dtype=torch.int8, interface="axi_stream_out")

    def generate_inputs(self, seed=None):
        self.data_in.fill_random()

    def forward(self, **inputs):
        return self.data_in.data


def _make_fmapio_spec() -> KernelSpec:
    """NPU fmapIO sub-kernel spec with AXI4-Lite ctrl + AXI4 DDR."""
    return KernelSpec(
        kernel_name="fmapIO",
        rtl_top="design/fmapIO/rtl/fmapIO_top.sv",
        parameters={
            "IN_DEPTH": "${IN_DEPTH}",
            "IN_HEIGHT": "${IN_HEIGHT}",
            "IN_WIDTH": "${IN_WIDTH}",
            "IN_CH": "${IN_CH}",
            "OUT_CH": "${OUT_CH}",
        },
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
                        name="in_ch",
                        offset=0x020,
                        auto_bind=AutoBindSpec(param="${IN_CH}"),
                    ),
                    RegisterSpec(
                        name="ifm_addr_lsb",
                        offset=0x038,
                        auto_bind=AutoBindSpec(
                            tensor="ifm", value="address", bits="31:0"
                        ),
                    ),
                    RegisterSpec(
                        name="vsync",
                        offset=0x050,
                        fields={"trigger": "0:0"},
                    ),
                    RegisterSpec(
                        name="layer_done",
                        offset=0x054,
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


class FmapIOKernel(Kernel):
    ifm = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ddr", direction=Direction.HOST_TO_DEV)
    ofm = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="ddr", direction=Direction.DEV_TO_HOST)

    def generate_inputs(self, seed=None):
        self.ifm.fill_random()


def _make_kernel_instance(
    kernel_class, spec, project_params=None, runtime_params=None
):
    """Helper to create and initialize a KernelInstance."""
    from vten.runtime.flattener import KernelInstance

    inst = KernelInstance(
        name=kernel_class.__name__,
        spec=spec,
        kernel_class=kernel_class,
        runtime_params=runtime_params or {},
    )
    inst.initialize(project_params or {})
    return inst


def _make_flat_view(kernel_instance, mappings=None, exposed=None):
    """Build FlattenedKernelView for a Unit kernel."""
    from vten.runtime.flattener import (
        ExposedTensor,
        FlattenedKernelView,
        InterfaceMapping,
    )

    if mappings is None:
        mappings = []
        for iface in kernel_instance.spec.interfaces.values():
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

    if exposed is None:
        exposed = {}
        for tensor in kernel_instance.tensors():
            exposed[tensor.name] = ExposedTensor(
                name=tensor.name,
                origin_path=f"_self.{tensor.name}",
                origin_tensor=tensor,
                top_interface=tensor.interface,
                direction=Direction.HOST_TO_DEV,
            )

    return FlattenedKernelView(
        name=kernel_instance.name,
        top_spec=kernel_instance.spec,
        sub_kernels={"_self": kernel_instance},
        interface_mappings=mappings,
        exposed_tensors=exposed,
        probe_points=[],
        connections=[],
    )


# ═══════════════════════════════════════════════════════════════════
# §7.3 — KernelInstance
# ═══════════════════════════════════════════════════════════════════


class TestKernelInstance:
    """KernelInstance initialization and attribute delegation."""

    def test_initialize_creates_instance(self):
        """initialize() creates kernel_class_instance."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        assert inst.kernel_class_instance is not None
        assert isinstance(inst.kernel_class_instance, PassthroughKernel)

    def test_initialize_resolves_parameters(self):
        """initialize() creates a ParameterResolver."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        assert inst._resolver is not None
        assert inst._resolver.resolve("${SIZE}") == 16

    def test_initialize_resolves_tensor_shapes(self):
        """Tensors get their parametric shapes resolved during initialize()."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 32}
        )
        tensors = inst.tensors()
        for t in tensors:
            assert t._resolved_shape is not None
            assert t._element_count == 32

    def test_initialize_shallow_copies_tensors(self):
        """Each instance gets shallow-copied tensors (not class originals)."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        # Instance tensor should have resolved shape, class descriptor should not
        class_tensor = PassthroughKernel._tensor_descriptors["data_in"]
        instance_tensor = inst.get_tensor("data_in")
        # They are different objects (shallow copy)
        assert instance_tensor is not class_tensor

    def test_tensors_returns_all(self):
        """tensors() returns all tensor descriptors."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        names = [t.name for t in inst.tensors()]
        assert "data_in" in names
        assert "data_out" in names

    def test_get_tensor_by_name(self):
        """get_tensor(name) returns the correct tensor."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        t = inst.get_tensor("data_in")
        assert t.name == "data_in"
        assert t.dtype == torch.int8

    def test_get_tensor_not_found(self):
        """get_tensor(name) raises AttributeError for unknown name."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        with pytest.raises(AttributeError):
            inst.get_tensor("nonexistent")

    def test_getattr_delegates_to_instance(self):
        """__getattr__ delegates to kernel_class_instance."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        # Accessing resolved param attribute through delegation
        t = inst.data_in
        assert isinstance(t, Tensor)
        assert t._resolved_shape == (8,)

    def test_getattr_raises_for_uninitialized(self):
        """__getattr__ raises AttributeError when not initialized."""
        from vten.runtime.flattener import KernelInstance

        inst = KernelInstance(
            name="test",
            spec=_make_passthrough_spec(),
            kernel_class=PassthroughKernel,
        )
        with pytest.raises(AttributeError, match="not initialized"):
            _ = inst.data_in

    def test_resolved_param_as_attribute(self):
        """Resolved parameters exposed as instance attributes."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 64}
        )
        # SIZE should be available on the kernel_class_instance
        assert inst.kernel_class_instance.SIZE == 64

    def test_project_params_override(self):
        """Project params provide defaults, runtime_params override."""
        inst = _make_kernel_instance(
            PassthroughKernel,
            _make_passthrough_spec(),
            project_params={"SIZE": 100},
            runtime_params={"SIZE": 16},
        )
        assert inst._resolver.resolve("${SIZE}") == 16


# ═══════════════════════════════════════════════════════════════════
# §7.1 — ExposedTensor
# ═══════════════════════════════════════════════════════════════════


class TestExposedTensor:
    """ExposedTensor property delegation and address management."""

    @pytest.fixture()
    def exposed_with_data(self):
        """ExposedTensor wrapping a resolved tensor with data."""
        from vten.runtime.flattener import ExposedTensor

        t = Tensor(shape=(4,), dtype=torch.int8, interface="test_iface")
        t.name = "test_tensor"
        t._resolved_shape = (4,)
        t._element_count = 4
        t.data = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        return ExposedTensor(
            name="test_tensor",
            origin_path="_self.test_tensor",
            origin_tensor=t,
            top_interface="test_iface",
            direction=Direction.HOST_TO_DEV,
        )

    def test_data_property_delegates(self, exposed_with_data):
        """data property returns origin_tensor.data."""
        assert torch.equal(
            exposed_with_data.data,
            torch.tensor([1, 2, 3, 4], dtype=torch.int8),
        )

    def test_data_setter_delegates(self, exposed_with_data):
        """data setter updates origin_tensor.data."""
        new_data = torch.tensor([5, 6, 7, 8], dtype=torch.int8)
        exposed_with_data.data = new_data
        assert torch.equal(exposed_with_data.origin_tensor.data, new_data)

    def test_shape_property(self, exposed_with_data):
        """shape property returns origin_tensor._resolved_shape."""
        assert exposed_with_data.shape == (4,)

    def test_element_count_property(self, exposed_with_data):
        """element_count returns origin_tensor._element_count."""
        assert exposed_with_data.element_count == 4

    def test_address_initially_none(self, exposed_with_data):
        """address is None before allocation."""
        assert exposed_with_data.address is None

    def test_set_address(self, exposed_with_data):
        """set_address() stores address on origin_tensor._address."""
        exposed_with_data.set_address(0x1000)
        assert exposed_with_data.address == 0x1000
        assert exposed_with_data.origin_tensor._address == 0x1000

    def test_serialized_initially_none(self, exposed_with_data):
        """_serialized is None before serialization stage."""
        assert exposed_with_data._serialized is None
        assert exposed_with_data._serialized_size == 0

    def test_port_buffers_initially_none(self, exposed_with_data):
        """_port_buffers is None for non-split/non-array interfaces."""
        assert exposed_with_data._port_buffers is None


# ═══════════════════════════════════════════════════════════════════
# §7.2 — InterfaceMapping
# ═══════════════════════════════════════════════════════════════════


class TestInterfaceMapping:
    """InterfaceMapping creation and MappingType variants."""

    def test_external_mapping(self):
        from vten.runtime.flattener import InterfaceMapping

        m = InterfaceMapping(
            sub_kernel="_self",
            sub_interface="axi_stream_in",
            mapping_type=MappingType.EXTERNAL,
            top_interface="axi_stream_in",
            bank_name=None,
            bank_offset=0,
        )
        assert m.mapping_type == MappingType.EXTERNAL
        assert m.bank_offset == 0

    def test_external_bank_mapping(self):
        """EXTERNAL_BANK with non-zero bank_offset (NPU composite)."""
        from vten.runtime.flattener import InterfaceMapping

        m = InterfaceMapping(
            sub_kernel="fmapio",
            sub_interface="ctrl",
            mapping_type=MappingType.EXTERNAL_BANK,
            top_interface="ctrl",
            bank_name="fmapio",
            bank_offset=0x0000,
        )
        assert m.mapping_type == MappingType.EXTERNAL_BANK

    def test_internal_mapping(self):
        from vten.runtime.flattener import InterfaceMapping

        m = InterfaceMapping(
            sub_kernel="fmapio",
            sub_interface="ifm_out",
            mapping_type=MappingType.INTERNAL,
            top_interface=None,
            bank_name=None,
            bank_offset=0,
        )
        assert m.mapping_type == MappingType.INTERNAL
        assert m.top_interface is None

    def test_internal_probe_mapping(self):
        from vten.runtime.flattener import InterfaceMapping

        m = InterfaceMapping(
            sub_kernel="fmapio",
            sub_interface="ifm_out",
            mapping_type=MappingType.INTERNAL_PROBE,
            top_interface=None,
            bank_name=None,
            bank_offset=0,
        )
        assert m.mapping_type == MappingType.INTERNAL_PROBE


# ═══════════════════════════════════════════════════════════════════
# §7.4 — FlattenedKernelView
# ═══════════════════════════════════════════════════════════════════


class TestFlattenedKernelView:
    """FlattenedKernelView query methods."""

    @pytest.fixture()
    def passthrough_view(self):
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        return _make_flat_view(inst)

    def test_external_interfaces_unit(self, passthrough_view):
        """Unit kernel: all interfaces are EXTERNAL."""
        ext = passthrough_view.external_interfaces()
        assert "axi_stream_in" in ext
        assert "axi_stream_out" in ext

    def test_external_interfaces_no_duplicates(self, passthrough_view):
        """external_interfaces() returns unique names."""
        ext = passthrough_view.external_interfaces()
        assert len(ext) == len(set(ext))

    def test_external_interfaces_excludes_internal(self):
        """INTERNAL mappings are not returned by external_interfaces()."""
        from vten.runtime.flattener import (
            FlattenedKernelView,
            InterfaceMapping,
        )

        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        mappings = [
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface="axi_stream_in",
                mapping_type=MappingType.EXTERNAL,
                top_interface="axi_stream_in",
                bank_name=None,
                bank_offset=0,
            ),
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface="internal_axis",
                mapping_type=MappingType.INTERNAL,
                top_interface=None,
                bank_name=None,
                bank_offset=0,
            ),
        ]
        view = FlattenedKernelView(
            name="test",
            top_spec=inst.spec,
            sub_kernels={"_self": inst},
            interface_mappings=mappings,
            exposed_tensors={},
            probe_points=[],
            connections=[],
        )
        ext = view.external_interfaces()
        assert "axi_stream_in" in ext
        assert None not in ext

    def test_tensors_for_interface(self, passthrough_view):
        """tensors_for_interface() returns all exposed tensors for that interface."""
        tensors = passthrough_view.tensors_for_interface("axi_stream_in")
        names = [t.name for t in tensors]
        assert "data_in" in names

    def test_tensors_for_interface_empty(self, passthrough_view):
        """Non-existent interface returns empty list."""
        tensors = passthrough_view.tensors_for_interface("nonexistent")
        assert tensors == []

    def test_registers_for_interface(self):
        """registers_for_interface() returns (sub_name, reg_spec, abs_offset)."""
        inst = _make_kernel_instance(
            FmapIOKernel,
            _make_fmapio_spec(),
            runtime_params={"IN_DEPTH": 1, "IN_HEIGHT": 8, "IN_WIDTH": 8, "IN_CH": 32, "OUT_CH": 32},
        )
        view = _make_flat_view(inst)
        regs = view.registers_for_interface("ctrl")
        assert len(regs) > 0
        # Each entry is (sub_kernel_name, RegisterSpec, absolute_offset)
        for sub_name, reg_spec, abs_offset in regs:
            assert sub_name == "_self"
            assert isinstance(reg_spec, RegisterSpec)
            assert abs_offset == reg_spec.offset  # bank_offset=0

    def test_registers_for_interface_with_bank_offset(self):
        """Bank offset is added to register offset."""
        from vten.runtime.flattener import (
            ExposedTensor,
            FlattenedKernelView,
            InterfaceMapping,
        )

        inst = _make_kernel_instance(
            FmapIOKernel,
            _make_fmapio_spec(),
            runtime_params={"IN_DEPTH": 1, "IN_HEIGHT": 8, "IN_WIDTH": 8, "IN_CH": 32, "OUT_CH": 32},
        )
        mappings = [
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface="ctrl",
                mapping_type=MappingType.EXTERNAL_BANK,
                top_interface="ctrl",
                bank_name="fmapio",
                bank_offset=0x1000,
            ),
        ]
        view = FlattenedKernelView(
            name="test",
            top_spec=inst.spec,
            sub_kernels={"_self": inst},
            interface_mappings=mappings,
            exposed_tensors={},
            probe_points=[],
            connections=[],
        )
        regs = view.registers_for_interface("ctrl")
        # e.g., "in_ch" offset 0x020 + bank_offset 0x1000 = 0x1020
        in_ch_reg = [r for r in regs if r[1].name == "in_ch"]
        assert len(in_ch_reg) == 1
        assert in_ch_reg[0][2] == 0x1000 + 0x020

    def test_resolve_auto_bind_tensor(self):
        """resolve_auto_bind_tensor() finds ExposedTensor by origin_path."""
        from vten.runtime.flattener import ExposedTensor

        inst = _make_kernel_instance(
            FmapIOKernel,
            _make_fmapio_spec(),
            runtime_params={"IN_DEPTH": 1, "IN_HEIGHT": 8, "IN_WIDTH": 8, "IN_CH": 32, "OUT_CH": 32},
        )
        view = _make_flat_view(inst)
        exposed = view.resolve_auto_bind_tensor("_self", "ifm")
        assert exposed.name == "ifm"
        assert exposed.origin_path == "_self.ifm"

    def test_resolve_auto_bind_tensor_not_found(self):
        """resolve_auto_bind_tensor() raises BindingError for missing tensor."""
        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 8}
        )
        view = _make_flat_view(inst)
        with pytest.raises(BindingError, match="no matching exposed tensor"):
            view.resolve_auto_bind_tensor("_self", "nonexistent")


# ═══════════════════════════════════════════════════════════════════
# §5 — Engine._wrap_unit_as_flat (Unit kernel wrapping)
# ═══════════════════════════════════════════════════════════════════


class TestWrapUnitAsFlat:
    """RuntimeEngine._wrap_unit_as_flat() creates correct view."""

    def test_unit_wrapping_creates_self_subkernel(self):
        """Unit kernel is wrapped with sub_kernel name '_self'."""
        from vten.runtime.engine import RuntimeEngine

        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        engine = RuntimeEngine(
            kernels={"PassthroughKernel": inst},
            ops=[],
            project_params={},
        )
        view = engine._wrap_unit_as_flat(inst)
        assert "_self" in view.sub_kernels
        assert view.sub_kernels["_self"] is inst

    def test_unit_wrapping_creates_external_mappings(self):
        """All interfaces become EXTERNAL mappings."""
        from vten.runtime.engine import RuntimeEngine

        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        engine = RuntimeEngine(
            kernels={"PassthroughKernel": inst},
            ops=[],
            project_params={},
        )
        view = engine._wrap_unit_as_flat(inst)
        for m in view.interface_mappings:
            assert m.mapping_type == MappingType.EXTERNAL
            assert m.sub_kernel == "_self"

    def test_unit_wrapping_exposes_all_tensors(self):
        """All kernel tensors become exposed tensors."""
        from vten.runtime.engine import RuntimeEngine

        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        engine = RuntimeEngine(
            kernels={"PassthroughKernel": inst},
            ops=[],
            project_params={},
        )
        view = engine._wrap_unit_as_flat(inst)
        assert "data_in" in view.exposed_tensors
        assert "data_out" in view.exposed_tensors

    def test_exposed_tensor_origin_path(self):
        """Exposed tensor origin_path is '_self.<tensor_name>'."""
        from vten.runtime.engine import RuntimeEngine

        inst = _make_kernel_instance(
            PassthroughKernel, _make_passthrough_spec(), runtime_params={"SIZE": 16}
        )
        engine = RuntimeEngine(
            kernels={"PassthroughKernel": inst},
            ops=[],
            project_params={},
        )
        view = engine._wrap_unit_as_flat(inst)
        assert view.exposed_tensors["data_in"].origin_path == "_self.data_in"


# ═══════════════════════════════════════════════════════════════════
# NPU 3D — FlattenedKernelView with multiple sub-kernels
# ═══════════════════════════════════════════════════════════════════


class TestNPU3DFlattenedView:
    """NPU 3D composite view with banked registers."""

    @pytest.fixture()
    def npu_view(self):
        """Simulate NPU 3D composite with fmapIO + bias_loader sub-kernels."""
        from vten.runtime.flattener import (
            ExposedTensor,
            FlattenedKernelView,
            InterfaceMapping,
        )

        # fmapIO sub-kernel
        fmapio_inst = _make_kernel_instance(
            FmapIOKernel,
            _make_fmapio_spec(),
            runtime_params={
                "IN_DEPTH": 1, "IN_HEIGHT": 8, "IN_WIDTH": 8,
                "IN_CH": 32, "OUT_CH": 32,
            },
        )

        # Create a composite-like spec with bank offsets
        composite_spec = KernelSpec(
            kernel_name="NPU3D",
            rtl_top="design/NPU_3D_top.sv",
            memory_regions={
                "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000, alignment=4096),
            },
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl",
                    rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L,
                    addr_width=16,
                    registers=list(fmapio_inst.spec.interfaces["ctrl"].registers),
                    register_banks=[
                        RegisterBankSpec(name="fmapio", base_offset=0x0000),
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

        # Interface mappings with bank offset
        mappings = [
            InterfaceMapping(
                sub_kernel="fmapio",
                sub_interface="ctrl",
                mapping_type=MappingType.EXTERNAL_BANK,
                top_interface="ctrl",
                bank_name="fmapio",
                bank_offset=0x0000,
            ),
            InterfaceMapping(
                sub_kernel="fmapio",
                sub_interface="ddr",
                mapping_type=MappingType.EXTERNAL,
                top_interface="ddr",
                bank_name=None,
                bank_offset=0,
            ),
        ]

        # Exposed tensors
        ifm_tensor = fmapio_inst.get_tensor("ifm")
        ofm_tensor = fmapio_inst.get_tensor("ofm")
        exposed = {
            "ifm": ExposedTensor(
                name="ifm",
                origin_path="fmapio.ifm",
                origin_tensor=ifm_tensor,
                top_interface="ddr",
                direction=Direction.HOST_TO_DEV,
            ),
            "ofm": ExposedTensor(
                name="ofm",
                origin_path="fmapio.ofm",
                origin_tensor=ofm_tensor,
                top_interface="ddr",
                direction=Direction.DEV_TO_HOST,
            ),
        }

        return FlattenedKernelView(
            name="NPU3D",
            top_spec=composite_spec,
            sub_kernels={"fmapio": fmapio_inst},
            interface_mappings=mappings,
            exposed_tensors=exposed,
            probe_points=[],
            connections=[],
        )

    def test_external_interfaces(self, npu_view):
        ext = npu_view.external_interfaces()
        assert "ctrl" in ext
        assert "ddr" in ext

    def test_registers_include_bank_offset(self, npu_view):
        """Registers for banked interface include bank_offset."""
        regs = npu_view.registers_for_interface("ctrl")
        # bank_offset=0x0000, so absolute = 0x0000 + register.offset
        in_ch = [r for r in regs if r[1].name == "in_ch"]
        assert len(in_ch) == 1
        assert in_ch[0][2] == 0x0000 + 0x020

    def test_resolve_auto_bind_tensor_composite(self, npu_view):
        """resolve_auto_bind_tensor with composite sub-kernel name."""
        exposed = npu_view.resolve_auto_bind_tensor("fmapio", "ifm")
        assert exposed.name == "ifm"
        assert exposed.origin_path == "fmapio.ifm"

    def test_tensors_for_ddr_interface(self, npu_view):
        """tensors_for_interface('ddr') returns ifm and ofm."""
        tensors = npu_view.tensors_for_interface("ddr")
        names = {t.name for t in tensors}
        assert names == {"ifm", "ofm"}

    def test_no_tensors_for_ctrl(self, npu_view):
        """AXI4-Lite ctrl has no exposed tensors."""
        tensors = npu_view.tensors_for_interface("ctrl")
        assert tensors == []
