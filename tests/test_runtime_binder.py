"""Phase 2 tests — Stage 5: auto_bind Resolution.

Spec reference: 02_runtime_engine.md §11, 00_data_models.md §5.3
NPU 3D patterns: npu_3d_analysis.md §4

Tests parse_bit_range(), resolve_auto_binds(), _compute_auto_bind_value()
for address bit-slicing, size_bytes, size_beats, size_elements, param, expr.
"""

from __future__ import annotations

import pytest
import torch

from vten.errors import BindingError
from vten.kernel.base import Kernel
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
    RegisterSpec,
)


# ── Helpers ──────────────────────────────────────────────────────


class SimpleKernel(Kernel):
    ifm = Tensor(shape=(32,), dtype=torch.int8, interface="ddr", direction=Direction.HOST_TO_DEV)
    ofm = Tensor(shape=(32,), dtype=torch.int8, interface="ddr", direction=Direction.DEV_TO_HOST)


def _make_simple_spec(registers=None) -> KernelSpec:
    """Spec with AXI4-Lite ctrl and AXI4 DDR interfaces."""
    if registers is None:
        registers = []
    return KernelSpec(
        kernel_name="simple",
        rtl_top="rtl/simple.sv",
        parameters={"SIZE": "${SIZE}", "IN_CH": "${IN_CH}"},
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


def _make_view_with_registers(registers, runtime_params=None):
    """Build FlattenedKernelView with given registers on 'ctrl' interface."""
    from vten.runtime.flattener import (
        ExposedTensor,
        FlattenedKernelView,
        InterfaceMapping,
        KernelInstance,
    )

    params = {"SIZE": 32, "IN_CH": 64}
    if runtime_params:
        params.update(runtime_params)

    spec = _make_simple_spec(registers)
    inst = KernelInstance(
        name="SimpleKernel",
        spec=spec,
        kernel_class=SimpleKernel,
        runtime_params=params,
    )
    inst.initialize({})

    mappings = [
        InterfaceMapping(
            sub_kernel="_self",
            sub_interface="ctrl",
            mapping_type=MappingType.EXTERNAL,
            top_interface="ctrl",
            bank_name=None,
            bank_offset=0,
        ),
        InterfaceMapping(
            sub_kernel="_self",
            sub_interface="ddr",
            mapping_type=MappingType.EXTERNAL,
            top_interface="ddr",
            bank_name=None,
            bank_offset=0,
        ),
    ]

    # Build exposed tensors with known state
    ifm_tensor = inst.get_tensor("ifm")
    ofm_tensor = inst.get_tensor("ofm")

    exposed = {
        "ifm": ExposedTensor(
            name="ifm",
            origin_path="_self.ifm",
            origin_tensor=ifm_tensor,
            top_interface="ddr",
            direction=Direction.HOST_TO_DEV,
        ),
        "ofm": ExposedTensor(
            name="ofm",
            origin_path="_self.ofm",
            origin_tensor=ofm_tensor,
            top_interface="ddr",
            direction=Direction.DEV_TO_HOST,
        ),
    }

    return FlattenedKernelView(
        name="SimpleKernel",
        top_spec=spec,
        sub_kernels={"_self": inst},
        interface_mappings=mappings,
        exposed_tensors=exposed,
        probe_points=[],
        connections=[],
    )


# ═══════════════════════════════════════════════════════════════════
# §11.1 — parse_bit_range
# ═══════════════════════════════════════════════════════════════════


class TestParseBitRange:
    """parse_bit_range("hi:lo") → (hi, lo) with validation."""

    @pytest.fixture()
    def parse(self):
        from vten.runtime.binder import parse_bit_range
        return parse_bit_range

    def test_lower_32_bits(self, parse):
        assert parse("31:0") == (31, 0)

    def test_upper_32_bits(self, parse):
        assert parse("63:32") == (63, 32)

    def test_single_bit(self, parse):
        assert parse("0:0") == (0, 0)

    def test_nibble_range(self, parse):
        assert parse("3:0") == (3, 0)

    def test_whitespace_stripped(self, parse):
        assert parse(" 15 : 8 ") == (15, 8)

    def test_hi_less_than_lo_raises(self, parse):
        with pytest.raises(ValueError, match="hi.*lo"):
            parse("0:7")

    def test_invalid_format_no_colon(self, parse):
        with pytest.raises(ValueError, match="hi:lo"):
            parse("31")

    def test_invalid_format_too_many_parts(self, parse):
        with pytest.raises(ValueError):
            parse("31:16:0")

    def test_non_numeric_raises(self, parse):
        with pytest.raises(ValueError):
            parse("abc:0")


# ═══════════════════════════════════════════════════════════════════
# §11.2 — _compute_auto_bind_value: address
# ═══════════════════════════════════════════════════════════════════


class TestAutoBindAddress:
    """auto_bind value='address' with optional bit slicing."""

    def test_full_address(self):
        """auto_bind address without bits → full address value."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="addr_full",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="address"),
            ),
        ]
        view = _make_view_with_registers(regs)
        # Set address on exposed tensor
        view.exposed_tensors["ifm"].set_address(0x0000_1000_DEAD_BEEF)

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 0x0000_1000_DEAD_BEEF

    def test_address_lower_32_bits(self):
        """auto_bind address bits='31:0' → lower 32 bits."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="addr_lsb",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="31:0"),
            ),
        ]
        view = _make_view_with_registers(regs)
        view.exposed_tensors["ifm"].set_address(0x0000_1000_DEAD_BEEF)

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 0xDEAD_BEEF

    def test_address_upper_32_bits(self):
        """auto_bind address bits='63:32' → upper 32 bits."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="addr_msb",
                offset=0x014,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="63:32"),
            ),
        ]
        view = _make_view_with_registers(regs)
        view.exposed_tensors["ifm"].set_address(0x0000_1000_DEAD_BEEF)

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 0x0000_1000

    def test_address_none_raises(self):
        """auto_bind address on stream tensor (no address) → BindingError."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="addr",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="address"),
            ),
        ]
        view = _make_view_with_registers(regs)
        # Don't set address → stays None
        with pytest.raises(BindingError, match="no address"):
            _compute_auto_bind_value(regs[0].auto_bind, "_self", view)

    def test_address_bit_slice_zero_address(self):
        """Address 0x0 with bits='31:0' → 0."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="addr_lsb",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="31:0"),
            ),
        ]
        view = _make_view_with_registers(regs)
        view.exposed_tensors["ifm"].set_address(0x0)

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 0


# ═══════════════════════════════════════════════════════════════════
# §11.3 — _compute_auto_bind_value: size variants
# ═══════════════════════════════════════════════════════════════════


class TestAutoBindSize:
    """auto_bind value='size_bytes', 'size_beats', 'size_elements'."""

    def test_size_bytes(self):
        """size_bytes returns exposed._serialized_size."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="buf_size",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="size_bytes"),
            ),
        ]
        view = _make_view_with_registers(regs)
        view.exposed_tensors["ifm"]._serialized_size = 1024

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 1024

    def test_size_beats(self):
        """size_beats = serialized_size / (data_width / 8)."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="num_beats",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="size_beats"),
            ),
        ]
        view = _make_view_with_registers(regs)
        # 256-bit bus → 32 bytes per beat
        # 1024 bytes → 32 beats
        view.exposed_tensors["ifm"]._serialized_size = 1024

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 32  # 1024 / (256/8)

    def test_size_elements(self):
        """size_elements returns exposed.element_count."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="elem_count",
                offset=0x010,
                auto_bind=AutoBindSpec(tensor="ifm", value="size_elements"),
            ),
        ]
        view = _make_view_with_registers(regs)

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 32  # shape=(32,)


# ═══════════════════════════════════════════════════════════════════
# §11.4 — _compute_auto_bind_value: param / expr
# ═══════════════════════════════════════════════════════════════════


class TestAutoBindParam:
    """auto_bind with param or expr (resolved via ParameterResolver)."""

    def test_param_resolution(self):
        """auto_bind param='${IN_CH}' → resolved value."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="in_ch",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${IN_CH}"),
            ),
        ]
        view = _make_view_with_registers(regs, runtime_params={"SIZE": 32, "IN_CH": 64})

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 64

    def test_expr_resolution(self):
        """auto_bind expr='${IN_CH} * 2' → evaluated expression."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="double_ch",
                offset=0x010,
                auto_bind=AutoBindSpec(expr="${IN_CH} * 2"),
            ),
        ]
        view = _make_view_with_registers(regs, runtime_params={"SIZE": 32, "IN_CH": 64})

        result = _compute_auto_bind_value(
            regs[0].auto_bind, "_self", view
        )
        assert result == 128

    def test_no_resolvable_value_raises(self):
        """auto_bind with no recognized value/param/expr → BindingError."""
        from vten.runtime.binder import _compute_auto_bind_value

        regs = [
            RegisterSpec(
                name="bad",
                offset=0x010,
                auto_bind=AutoBindSpec(),  # all None
            ),
        ]
        view = _make_view_with_registers(regs)

        with pytest.raises(BindingError, match="no resolvable value"):
            _compute_auto_bind_value(regs[0].auto_bind, "_self", view)


# ═══════════════════════════════════════════════════════════════════
# §11.5 — resolve_auto_binds (full integration)
# ═══════════════════════════════════════════════════════════════════


class TestResolveAutoBinds:
    """resolve_auto_binds() processes all auto_bind registers."""

    def test_returns_binding_entries(self):
        """resolve_auto_binds returns list of RegisterBindingEntry."""
        from vten.runtime.binder import RegisterBindingEntry, resolve_auto_binds

        regs = [
            RegisterSpec(
                name="in_ch",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${IN_CH}"),
            ),
        ]
        view = _make_view_with_registers(regs)
        bindings = resolve_auto_binds(view)
        assert len(bindings) == 1
        assert isinstance(bindings[0], RegisterBindingEntry)

    def test_binding_entry_fields(self):
        """RegisterBindingEntry has correct register_name and resolved_value."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="in_ch",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${IN_CH}"),
            ),
        ]
        view = _make_view_with_registers(regs)
        bindings = resolve_auto_binds(view)
        b = bindings[0]
        assert b.register_name == "_self.in_ch"
        assert b.resolved_value == 64
        assert b.absolute_offset == 0x010
        assert b.interface_name == "ctrl"

    def test_skips_non_auto_bind(self):
        """Registers without auto_bind are skipped."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(name="vsync", offset=0x050, fields={"trigger": "0:0"}),
            RegisterSpec(
                name="in_ch",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${IN_CH}"),
            ),
        ]
        view = _make_view_with_registers(regs)
        bindings = resolve_auto_binds(view)
        # Only in_ch has auto_bind
        assert len(bindings) == 1
        assert bindings[0].register_name == "_self.in_ch"

    def test_npu_3d_address_split_bindings(self):
        """NPU 3D pattern: address LSB/MSB pair resolved correctly."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="ifm_addr_lsb",
                offset=0x038,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="31:0"),
            ),
            RegisterSpec(
                name="ifm_addr_msb",
                offset=0x03C,
                auto_bind=AutoBindSpec(tensor="ifm", value="address", bits="63:32"),
            ),
        ]
        view = _make_view_with_registers(regs)
        view.exposed_tensors["ifm"].set_address(0x0000_0001_8000_0000)

        bindings = resolve_auto_binds(view)
        assert len(bindings) == 2

        lsb_binding = [b for b in bindings if "lsb" in b.register_name][0]
        msb_binding = [b for b in bindings if "msb" in b.register_name][0]

        assert lsb_binding.resolved_value == 0x8000_0000
        assert msb_binding.resolved_value == 0x0000_0001

    def test_multiple_param_bindings(self):
        """Multiple param auto_binds all resolved."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="size_reg",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${SIZE}"),
            ),
            RegisterSpec(
                name="ch_reg",
                offset=0x014,
                auto_bind=AutoBindSpec(param="${IN_CH}"),
            ),
        ]
        view = _make_view_with_registers(regs)
        bindings = resolve_auto_binds(view)
        assert len(bindings) == 2
        values = {b.register_name: b.resolved_value for b in bindings}
        assert values["_self.size_reg"] == 32
        assert values["_self.ch_reg"] == 64

    def test_kernel_path_format(self):
        """kernel_path is '{view_name}.{sub_name}.{interface_name}'."""
        from vten.runtime.binder import resolve_auto_binds

        regs = [
            RegisterSpec(
                name="reg_a",
                offset=0x010,
                auto_bind=AutoBindSpec(param="${SIZE}"),
            ),
        ]
        view = _make_view_with_registers(regs)
        bindings = resolve_auto_binds(view)
        assert bindings[0].kernel_path == "SimpleKernel._self.ctrl"
