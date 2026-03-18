"""Tests for vten.spec — kernel_spec.yaml parser and data models.

Spec reference: 03_kernel_spec_schema.md, 00_data_models.md §5.8
NPU 3D patterns: npu_3d_analysis.md §14 (test_spec_parser.py)

NPU 3D spec parsing patterns:
  - 6 AXI4-Lite interfaces (per-IP control registers)
  - AXI4 DDR: data_width=256, addr_width=64, tensors=[ifm,ofm,concat]
  - AXI4 HBM: split.mode=channel_interleave, 32 ports
  - auto_bind: address split, size_bytes, param binding
  - packing.bus_width: 8×32=256 exact match
  - register offset ranges: bias_loader 0x010~0x01C, weight_loader 0x010~0x128
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vten.spec.models import (
    KernelSpec,
    InterfaceSpec,
    PackingScheme,
    MemoryRegion,
    CustomField,
)
from vten.spec.parser import parse_kernel_spec
from vten.runtime.errors import SpecValidationError


# ── Helper ─────────────────────────────────────────────────────────


def _write_spec(tmp_path: Path, data: dict, name: str = "test.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return p


# ═══════════════════════════════════════════════════════════════════
# §1  Minimal valid specs — passthrough baseline
# ═══════════════════════════════════════════════════════════════════


class TestMinimalSpec:

    def test_axi4s_minimal(self, tmp_path):
        data = {
            "kernel": "passthrough",
            "rtl_top": "rtl/pt.sv",
            "interfaces": {
                "axis_in": {
                    "rtl_port": "s_axis",
                    "protocol": "axi4_stream",
                    "tensor": "data_in",
                    "packing": {
                        "element_width": 8,
                        "elements_per_beat": 4,
                    },
                },
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert isinstance(spec, KernelSpec)
        assert spec.kernel_name == "passthrough"
        assert spec.rtl_top == "rtl/pt.sv"
        assert "axis_in" in spec.interfaces

    def test_interface_spec_fields(self, tmp_path):
        data = {
            "kernel": "test",
            "rtl_top": "rtl/t.sv",
            "interfaces": {
                "s": {
                    "rtl_port": "s_axis",
                    "protocol": "axi4_stream",
                    "tensor": "t",
                    "packing": {"element_width": 16, "elements_per_beat": 2},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        iface = spec.get_interface("s")
        assert isinstance(iface, InterfaceSpec)
        assert iface.rtl_port == "s_axis"
        assert iface.tensor == "t"
        assert iface.packing is not None
        assert iface.packing.element_width == 16
        assert iface.packing.elements_per_beat == 2


# ═══════════════════════════════════════════════════════════════════
# §2  Parameters — NPU 3D 9-parameter fmapIO
# ═══════════════════════════════════════════════════════════════════


class TestParameters:

    def test_fmapio_variable_params(self, tmp_path):
        """fmapIO: 9개 파라미터 (모두 런타임 변수)."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "parameters": {
                "IN_DEPTH": "${IN_DEPTH}", "IN_HEIGHT": "${IN_HEIGHT}",
                "IN_WIDTH": "${IN_WIDTH}", "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}",
                "IFM_STRIDE": "${IFM_STRIDE}", "OFM_STRIDE": "${OFM_STRIDE}",
                "IS_CONCAT": "${IS_CONCAT}", "CONCAT_CH": "${CONCAT_CH}",
            },
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axilite_ctrl",
                    "protocol": "axi4_lite",
                    "registers": [
                        {"name": "in_depth", "offset": 0x014,
                         "auto_bind": {"param": "${IN_DEPTH}"}},
                    ],
                },
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert len(spec.parameters) == 9
        assert spec.parameters["IN_CH"] == "${IN_CH}"
        assert spec.parameters["OUT_CH"] == "${OUT_CH}"

    def test_weight_loader_mixed_params(self, tmp_path):
        """weight_loader: 6개 파라미터 (변수 + 고정 가능)."""
        data = {
            "kernel": "weight_loader",
            "rtl_top": "design/weight_loader/rtl/weight_loader_top.sv",
            "parameters": {
                "IN_WIDTH": "${IN_WIDTH}", "IN_HEIGHT": "${IN_HEIGHT}",
                "IN_DEPTH": "${IN_DEPTH}", "IN_CH": "${IN_CH}",
                "OUT_CH": "${OUT_CH}", "KERNEL_SIZE": "${KERNEL_SIZE}",
            },
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axilite_ctrl",
                    "protocol": "axi4_lite",
                    "registers": [
                        {"name": "in_width", "offset": 0x010,
                         "auto_bind": {"param": "${IN_WIDTH}"}},
                    ],
                },
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.parameters["KERNEL_SIZE"] == "${KERNEL_SIZE}"

    def test_no_parameters(self, tmp_path):
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "s": {
                    "rtl_port": "p",
                    "protocol": "axi4_stream",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.parameters == {}


# ═══════════════════════════════════════════════════════════════════
# §3  Memory regions — NPU 3D DDR + HBM
# ═══════════════════════════════════════════════════════════════════


class TestMemoryRegions:

    def test_ddr_region_4gb(self, tmp_path):
        """DDR region: 4GB, 4096-byte alignment (fmapIO/bias_loader)."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "memory_regions": {
                "ddr": {"base": 0, "size": 0x1_0000_0000, "alignment": 4096},
            },
            "interfaces": {
                "ddr": {
                    "rtl_port": "m_axi_ddr",
                    "protocol": "axi4",
                    "data_width": 256,
                    "addr_width": 64,
                    "memory_region": "ddr",
                    "tensors": ["ifm", "ofm", "concat"],
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert "ddr" in spec.memory_regions
        mr = spec.memory_regions["ddr"]
        assert isinstance(mr, MemoryRegion)
        assert mr.base == 0
        assert mr.size == 0x1_0000_0000
        assert mr.alignment == 4096

    def test_hbm_region(self, tmp_path):
        """HBM region: weight_loader 32-bank storage."""
        data = {
            "kernel": "weight_loader",
            "rtl_top": "design/weight_loader/rtl/weight_loader_top.sv",
            "memory_regions": {
                "hbm": {"base": 0, "size": 0x1_0000_0000},
            },
            "interfaces": {
                "hbm": {
                    "rtl_port": "m_axi_hbm",
                    "protocol": "axi4",
                    "data_width": 256,
                    "addr_width": 64,
                    "memory_region": "hbm",
                    "tensor": "weight",
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert "hbm" in spec.memory_regions

    def test_default_alignment(self, tmp_path):
        """alignment 미지정 → 기본 4096."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {
                "hbm": {"base": 0, "size": 0x1_0000_0000},
            },
            "interfaces": {
                "m": {
                    "rtl_port": "m",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "hbm",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.memory_regions["hbm"].alignment == 4096


# ═══════════════════════════════════════════════════════════════════
# §4  AXI4-Stream interface — NPU 3D internal streams
# ═══════════════════════════════════════════════════════════════════


class TestAXI4StreamInterface:

    def test_stream_protocol(self, passthrough_spec_yaml):
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        iface = spec.get_interface("axi_stream_in")
        from vten.spec.models import Protocol
        assert iface.protocol == Protocol.AXI4S

    def test_ifm_stream_256bit(self, tmp_path):
        """fmapIO ifm_out: 8-bit × 32 = 256-bit AXI-Stream."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "interfaces": {
                "ifm_out": {
                    "rtl_port": "m_axis_ifm",
                    "protocol": "axi4_stream",
                    "tensor": "ifm",
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                },
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        iface = spec.get_interface("ifm_out")
        assert iface.packing.element_width == 8
        assert iface.packing.elements_per_beat == 32
        assert iface.packing.bus_width == 256

    def test_bias_stream_int32(self, tmp_path):
        """bias_loader bias_out: 32-bit × 8 = 256-bit."""
        data = {
            "kernel": "bias_loader",
            "rtl_top": "design/bias_loader/rtl/bias_loader_top.sv",
            "interfaces": {
                "bias_out": {
                    "rtl_port": "m_axis_bias",
                    "protocol": "axi4_stream",
                    "tensor": "bias",
                    "packing": {"element_width": 32, "elements_per_beat": 8},
                },
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        iface = spec.get_interface("bias_out")
        assert iface.packing.bus_width == 256  # 32 × 8

    def test_stream_no_data_width(self, passthrough_spec_yaml):
        """AXI4-Stream에서는 data_width가 packing에서 유추된다."""
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        iface = spec.get_interface("axi_stream_in")
        if iface.data_width is not None:
            assert iface.data_width == iface.packing.bus_width


# ═══════════════════════════════════════════════════════════════════
# §5  AXI4 memory-mapped — NPU 3D DDR/HBM interfaces
# ═══════════════════════════════════════════════════════════════════


class TestAXI4Interface:

    def test_fmapio_ddr_full_parse(self, fmapio_spec_yaml):
        """fmapIO DDR: 256-bit data, 64-bit addr, tensors=[ifm, ofm, concat]."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        iface = spec.get_interface("ddr")
        from vten.spec.models import Protocol
        assert iface.protocol == Protocol.AXI4
        assert iface.data_width == 256
        assert iface.addr_width == 64
        assert iface.memory_region == "ddr"

    def test_fmapio_ddr_tensors_list(self, fmapio_spec_yaml):
        """fmapIO DDR: 3 tensors 공유 — ifm, ofm, concat."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        iface = spec.get_interface("ddr")
        assert iface.tensor is None
        assert iface.tensors == ["ifm", "ofm", "concat"]

    def test_bias_loader_ddr_single_tensor(self, bias_loader_spec_yaml):
        """bias_loader DDR: 단일 tensor (bias)."""
        spec = parse_kernel_spec(str(bias_loader_spec_yaml))
        iface = spec.get_interface("ddr")
        assert iface.tensor == "bias"

    def test_bias_loader_int32_packing(self, bias_loader_spec_yaml):
        """bias_loader: int32 packing — element_width=32, elements_per_beat=8."""
        spec = parse_kernel_spec(str(bias_loader_spec_yaml))
        iface = spec.get_interface("ddr")
        assert iface.packing.element_width == 32
        assert iface.packing.elements_per_beat == 8
        assert iface.packing.bus_width == 256

    def test_weight_loader_hbm_split(self, weight_loader_spec_yaml):
        """weight_loader HBM: 32-port channel_interleave split."""
        spec = parse_kernel_spec(str(weight_loader_spec_yaml))
        iface = spec.get_interface("hbm")
        assert iface.split is not None
        assert iface.split["mode"] == "channel_interleave"
        assert len(iface.split["ports"]) == 32

    def test_axi4_default_addr_width(self, tmp_path):
        """addr_width 미지정 시 기본값 64."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1000}},
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "ddr",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.get_interface("m").addr_width == 64

    def test_tensors_list_generic(self, tmp_path):
        """tensors (복수) 필드 파싱."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1000}},
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "ddr",
                    "tensors": ["ifm", "weight", "ofm"],
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        iface = spec.get_interface("m")
        assert iface.tensor is None
        assert iface.tensors == ["ifm", "weight", "ofm"]


# ═══════════════════════════════════════════════════════════════════
# §6  AXI4-Lite interface & registers — NPU 3D per-IP registers
# ═══════════════════════════════════════════════════════════════════


class TestAXI4LiteInterface:

    def test_fmapio_ctrl_registers(self, fmapio_spec_yaml):
        """fmapIO ctrl: 17 registers, 16-bit addr."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        iface = spec.get_interface("ctrl")
        from vten.spec.models import Protocol
        assert iface.protocol == Protocol.AXI4L
        assert iface.addr_width == 16
        assert iface.registers is not None
        assert len(iface.registers) == 17  # fmapIO has 17 regs in fixture

    def test_fmapio_register_offsets(self, fmapio_spec_yaml):
        """fmapIO: in_depth@0x014, vsync@0x050, layer_done@0x054."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        regs = spec.get_registers("ctrl")
        reg_map = {r.name: r for r in regs}
        assert reg_map["in_depth"].offset == 0x014
        assert reg_map["vsync"].offset == 0x050
        assert reg_map["layer_done"].offset == 0x054

    def test_fmapio_vsync_field(self, fmapio_spec_yaml):
        """fmapIO vsync: fields = {trigger: '0:0'}."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        regs = spec.get_registers("ctrl")
        vsync = next(r for r in regs if r.name == "vsync")
        assert vsync.fields is not None
        assert vsync.fields["trigger"] == "0:0"

    def test_fmapio_auto_bind_address_split(self, fmapio_spec_yaml):
        """fmapIO: ifm_addr split 64-bit → bits '31:0' + '63:32'."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        regs = spec.get_registers("ctrl")
        ifm_lo = next(r for r in regs if r.name == "ifm_addr_lsb")
        ifm_hi = next(r for r in regs if r.name == "ifm_addr_msb")
        assert ifm_lo.auto_bind is not None
        assert ifm_lo.auto_bind.tensor == "ifm"
        assert ifm_lo.auto_bind.value == "address"
        assert ifm_lo.auto_bind.bits == "31:0"
        assert ifm_hi.auto_bind.bits == "63:32"

    def test_fmapio_auto_bind_param(self, fmapio_spec_yaml):
        """fmapIO: in_ch register → auto_bind param '${IN_CH}'."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        regs = spec.get_registers("ctrl")
        in_ch = next(r for r in regs if r.name == "in_ch")
        assert in_ch.auto_bind is not None
        assert in_ch.auto_bind.param == "${IN_CH}"

    def test_bias_loader_register_range(self, bias_loader_spec_yaml):
        """bias_loader: 4 registers, offset 0x010~0x01C."""
        spec = parse_kernel_spec(str(bias_loader_spec_yaml))
        regs = spec.get_registers("ctrl")
        assert len(regs) == 4
        offsets = sorted(r.offset for r in regs)
        assert offsets[0] == 0x010
        assert offsets[-1] == 0x01C

    def test_weight_loader_register_count(self, weight_loader_spec_yaml):
        """weight_loader: 71 registers (3 params + 64 bank addrs + vsync + 3 params)."""
        spec = parse_kernel_spec(str(weight_loader_spec_yaml))
        regs = spec.get_registers("ctrl")
        assert len(regs) == 71  # 3 + 64 + 1 + 3

    def test_weight_loader_register_range(self, weight_loader_spec_yaml):
        """weight_loader: offset range 0x010~0x128."""
        spec = parse_kernel_spec(str(weight_loader_spec_yaml))
        regs = spec.get_registers("ctrl")
        offsets = sorted(r.offset for r in regs)
        assert offsets[0] == 0x010
        assert offsets[-1] == 0x128  # out_ch at 0x128

    def test_mac_atu_auto_bind_only(self, mac_atu_spec_yaml):
        """mac_atu ctrl: 8 registers, 모두 auto_bind (param)."""
        spec = parse_kernel_spec(str(mac_atu_spec_yaml))
        regs = spec.get_registers("ctrl")
        assert len(regs) == 8
        for reg in regs:
            assert reg.auto_bind is not None
            assert reg.auto_bind.param is not None

    def test_lite_default_addr_width(self, tmp_path):
        """AXI4-Lite addr_width 기본값 32."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axil",
                    "protocol": "axi4_lite",
                    "registers": [
                        {"name": "r0", "offset": 0},
                    ],
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec.get_interface("ctrl").addr_width == 32


# ═══════════════════════════════════════════════════════════════════
# §7  Register banks (CompositeKernel — NPU 3D 6-IP register banks)
# ═══════════════════════════════════════════════════════════════════


class TestRegisterBanks:

    def _npu_3d_composite_spec(self, tmp_path):
        """NPU 3D CompositeKernel: 6 register banks on unified ctrl."""
        data = {
            "kernel": "NPU_3D_top",
            "rtl_top": "design/NPU_3D_top.sv",
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axilite_ctrl",
                    "protocol": "axi4_lite",
                    "addr_width": 16,
                    "registers": [
                        {"name": "global_status", "offset": 0x000},
                    ],
                    "register_banks": {
                        "fmapio": {"base_offset": 0x000},
                        "bias_loader": {"base_offset": 0x100},
                        "weight_loader": {"base_offset": 0x200},
                        "mac_atu": {"base_offset": 0x400},
                        "psum_buffer": {"base_offset": 0x600},
                        "act_quant": {"base_offset": 0x700},
                    },
                },
            },
        }
        return _write_spec(tmp_path, data, "npu_3d_top.yaml")

    def test_six_register_banks(self, tmp_path):
        """NPU 3D: 6 register banks for 6 IP kernels."""
        spec = parse_kernel_spec(str(self._npu_3d_composite_spec(tmp_path)))
        iface = spec.get_interface("ctrl")
        assert iface.register_banks is not None
        bank_names = {b.name for b in iface.register_banks}
        assert bank_names == {
            "fmapio", "bias_loader", "weight_loader",
            "mac_atu", "psum_buffer", "act_quant",
        }

    def test_bank_offsets(self, tmp_path):
        """NPU 3D: bank offset 간 충분한 간격."""
        spec = parse_kernel_spec(str(self._npu_3d_composite_spec(tmp_path)))
        assert spec.get_bank_offset("ctrl", "fmapio") == 0x000
        assert spec.get_bank_offset("ctrl", "bias_loader") == 0x100
        assert spec.get_bank_offset("ctrl", "weight_loader") == 0x200
        assert spec.get_bank_offset("ctrl", "mac_atu") == 0x400

    def test_bank_not_found_raises(self, tmp_path):
        spec = parse_kernel_spec(str(self._npu_3d_composite_spec(tmp_path)))
        with pytest.raises(ValueError, match="Bank 'nonexistent' not found"):
            spec.get_bank_offset("ctrl", "nonexistent")


# ═══════════════════════════════════════════════════════════════════
# §8  PackingScheme — NPU 3D bus widths
# ═══════════════════════════════════════════════════════════════════


class TestPackingScheme:

    def test_npu_int8_256bit(self):
        """NPU 3D standard: 8-bit × 32 = 256-bit."""
        p = PackingScheme(element_width=8, elements_per_beat=32)
        assert p.bus_width == 256

    def test_npu_int32_256bit(self):
        """NPU 3D bias: 32-bit × 8 = 256-bit."""
        p = PackingScheme(element_width=32, elements_per_beat=8)
        assert p.bus_width == 256

    def test_passthrough_32bit(self):
        """Passthrough: 8-bit × 4 = 32-bit."""
        p = PackingScheme(element_width=8, elements_per_beat=4)
        assert p.bus_width == 32

    def test_bus_width_aligned(self):
        """aligned: 각 element를 byte boundary에 맞춤."""
        p = PackingScheme(
            element_width=12, elements_per_beat=2, alignment="aligned"
        )
        elem_bytes = (12 + 7) // 8  # = 2
        expected = elem_bytes * 8 * 2  # = 32
        assert p.bus_width == expected

    def test_defaults(self):
        p = PackingScheme(element_width=8, elements_per_beat=1)
        assert p.bit_order == "lsb_first"
        assert p.alignment == "packed"
        assert p.byte_order == "little"
        assert p.mode == "standard"

    def test_custom_mode_bus_width(self):
        p = PackingScheme(
            element_width=8,
            elements_per_beat=1,
            mode="custom",
            custom_fields=[
                CustomField(name="a", bits=(0, 7)),
                CustomField(name="b", bits=(8, 15)),
                CustomField(name="c", bits=(16, 31)),
            ],
        )
        assert p.bus_width == 32

    def test_custom_field_overlap_validation(self):
        p = PackingScheme(
            element_width=8,
            elements_per_beat=1,
            mode="custom",
            custom_fields=[
                CustomField(name="a", bits=(0, 7)),
                CustomField(name="b", bits=(4, 11)),  # overlaps with a
            ],
        )
        from vten.runtime.errors import ValidationError
        with pytest.raises(ValidationError, match="overlap"):
            p.validate_custom_fields()

    def test_custom_no_overlap_passes(self):
        p = PackingScheme(
            element_width=8,
            elements_per_beat=1,
            mode="custom",
            custom_fields=[
                CustomField(name="a", bits=(0, 7)),
                CustomField(name="b", bits=(8, 15)),
            ],
        )
        p.validate_custom_fields()  # should not raise


# ═══════════════════════════════════════════════════════════════════
# §9  Packing width constraint (AXI4) — NPU 3D 256-bit exact match
# ═══════════════════════════════════════════════════════════════════


class TestPackingWidthConstraint:

    def test_npu_exact_match_256bit(self, tmp_path):
        """NPU 3D: bus_width(256) == data_width(256) → OK."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1_0000_0000}},
            "interfaces": {
                "ddr": {
                    "rtl_port": "m_axi_ddr",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "ddr",
                    "tensors": ["ifm", "ofm"],
                    "packing": {
                        "element_width": 8,
                        "elements_per_beat": 32,  # 256 == 256
                    },
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        assert spec is not None

    def test_bus_width_exceeds_data_width_raises(self, tmp_path):
        """packing.bus_width > data_width → SpecValidationError."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1000}},
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 32,
                    "memory_region": "ddr",
                    "tensor": "t",
                    "packing": {
                        "element_width": 8,
                        "elements_per_beat": 8,  # bus_width = 64 > 32
                    },
                }
            },
        }
        with pytest.raises(SpecValidationError, match="exceeds data_width"):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_bus_width_less_than_data_width_warns(self, tmp_path):
        """packing.bus_width < data_width → warning (zero-padding)."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1000}},
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "ddr",
                    "tensor": "t",
                    "packing": {
                        "element_width": 8,
                        "elements_per_beat": 4,  # bus_width = 32 < 256
                    },
                }
            },
        }
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parse_kernel_spec(str(_write_spec(tmp_path, data)))
            assert len(w) >= 1
            assert "zero-padded" in str(w[0].message).lower() or "bus_width" in str(w[0].message)


# ═══════════════════════════════════════════════════════════════════
# §10  Validation errors
# ═══════════════════════════════════════════════════════════════════


class TestValidationErrors:

    def test_missing_kernel_field(self, tmp_path):
        data = {
            "rtl_top": "r.sv",
            "interfaces": {
                "s": {
                    "rtl_port": "p",
                    "protocol": "axi4_stream",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_missing_rtl_top(self, tmp_path):
        data = {
            "kernel": "k",
            "interfaces": {
                "s": {
                    "rtl_port": "p",
                    "protocol": "axi4_stream",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_missing_interfaces(self, tmp_path):
        data = {"kernel": "k", "rtl_top": "r.sv"}
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_empty_interfaces(self, tmp_path):
        data = {"kernel": "k", "rtl_top": "r.sv", "interfaces": {}}
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_missing_rtl_port(self, tmp_path):
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "s": {
                    "protocol": "axi4_stream",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_missing_protocol(self, tmp_path):
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "s": {
                    "rtl_port": "p",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_tensor_and_tensors_mutually_exclusive(self, tmp_path):
        """tensor와 tensors 동시 사용 불가."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "memory_regions": {"ddr": {"base": 0, "size": 0x1000}},
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "ddr",
                    "tensor": "t",
                    "tensors": ["t1", "t2"],
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_axi4_missing_memory_region_ref(self, tmp_path):
        """AXI4의 memory_region이 존재하지 않는 region을 참조."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "m": {
                    "rtl_port": "m_axi",
                    "protocol": "axi4",
                    "data_width": 256,
                    "memory_region": "nonexistent",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 32},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))

    def test_invalid_protocol_string(self, tmp_path):
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "interfaces": {
                "s": {
                    "rtl_port": "p",
                    "protocol": "invalid_protocol",
                    "tensor": "t",
                    "packing": {"element_width": 8, "elements_per_beat": 1},
                }
            },
        }
        with pytest.raises(SpecValidationError):
            parse_kernel_spec(str(_write_spec(tmp_path, data)))


# ═══════════════════════════════════════════════════════════════════
# §11  auto_bind validation — NPU 3D patterns
# ═══════════════════════════════════════════════════════════════════


class TestAutoBindValidation:

    def test_auto_bind_param_in_ch(self, tmp_path):
        """NPU 3D: in_ch register → auto_bind param ${IN_CH}."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "parameters": {"IN_CH": "${IN_CH}"},
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axilite_ctrl",
                    "protocol": "axi4_lite",
                    "registers": [
                        {
                            "name": "in_ch",
                            "offset": 0x020,
                            "auto_bind": {"param": "${IN_CH}"},
                        },
                    ],
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        reg = spec.get_registers("ctrl")[0]
        assert reg.auto_bind.param == "${IN_CH}"

    def test_auto_bind_address_split(self, tmp_path):
        """NPU 3D: ifm address → split into bits 31:0 and 63:32."""
        data = {
            "kernel": "fmapIO",
            "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axilite_ctrl",
                    "protocol": "axi4_lite",
                    "registers": [
                        {
                            "name": "ifm_addr_lsb",
                            "offset": 0x038,
                            "auto_bind": {"tensor": "ifm", "value": "address", "bits": "31:0"},
                        },
                        {
                            "name": "ifm_addr_msb",
                            "offset": 0x03C,
                            "auto_bind": {"tensor": "ifm", "value": "address", "bits": "63:32"},
                        },
                    ],
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        regs = spec.get_registers("ctrl")
        lsb = next(r for r in regs if r.name == "ifm_addr_lsb")
        msb = next(r for r in regs if r.name == "ifm_addr_msb")
        assert lsb.auto_bind.tensor == "ifm"
        assert lsb.auto_bind.value == "address"
        assert lsb.auto_bind.bits == "31:0"
        assert msb.auto_bind.bits == "63:32"

    def test_auto_bind_expr(self, tmp_path):
        """auto_bind expr: 산술 표현식."""
        data = {
            "kernel": "k",
            "rtl_top": "r.sv",
            "parameters": {"N": "${N}", "K": "${K}"},
            "interfaces": {
                "ctrl": {
                    "rtl_port": "s_axil",
                    "protocol": "axi4_lite",
                    "registers": [
                        {
                            "name": "total",
                            "offset": 0x30,
                            "auto_bind": {"expr": "${N}*${K}"},
                        },
                    ],
                }
            },
        }
        spec = parse_kernel_spec(str(_write_spec(tmp_path, data)))
        reg = spec.get_registers("ctrl")[0]
        assert reg.auto_bind.expr == "${N}*${K}"


# ═══════════════════════════════════════════════════════════════════
# §12  KernelSpec utility methods — NPU 3D fixtures
# ═══════════════════════════════════════════════════════════════════


class TestKernelSpecMethods:

    def test_get_interface(self, passthrough_spec_yaml):
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        iface = spec.get_interface("axi_stream_in")
        assert iface.name == "axi_stream_in"

    def test_get_interface_not_found(self, passthrough_spec_yaml):
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        with pytest.raises(KeyError, match="not found"):
            spec.get_interface("nonexistent")

    def test_fmapio_interface_names(self, fmapio_spec_yaml):
        """fmapIO: ctrl + ddr + ifm_out + ofm_in = 4 interfaces."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        names = spec.interface_names()
        assert set(names) == {"ctrl", "ddr", "ifm_out", "ofm_in"}

    def test_passthrough_interface_names(self, passthrough_spec_yaml):
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        names = spec.interface_names()
        assert set(names) == {"axi_stream_in", "axi_stream_out"}

    def test_get_registers_empty(self, passthrough_spec_yaml):
        spec = parse_kernel_spec(str(passthrough_spec_yaml))
        regs = spec.get_registers("axi_stream_in")
        assert regs == []

    def test_get_registers_fmapio(self, fmapio_spec_yaml):
        """fmapIO ctrl: 17 registers with named entries."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        regs = spec.get_registers("ctrl")
        assert len(regs) == 17
        names = {r.name for r in regs}
        assert "in_depth" in names
        assert "ifm_addr_lsb" in names
        assert "vsync" in names

    def test_fmapio_kernel_name(self, fmapio_spec_yaml):
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        assert spec.kernel_name == "fmapIO"

    def test_fmapio_memory_region_referenced(self, fmapio_spec_yaml):
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        assert spec.get_interface("ddr").memory_region == "ddr"
        assert "ddr" in spec.memory_regions


# ═══════════════════════════════════════════════════════════════════
# §13  Protocol enum values (00_data_models.md §1.1)
# ═══════════════════════════════════════════════════════════════════


class TestProtocolEnum:

    def test_protocol_values(self):
        from vten.spec.models import Protocol

        assert Protocol.AXI4S.value == "axi4_stream"
        assert Protocol.AXI4.value == "axi4"
        assert Protocol.AXI4L.value == "axi4_lite"

    def test_protocol_from_string(self):
        from vten.spec.models import Protocol

        assert Protocol("axi4_stream") == Protocol.AXI4S
        assert Protocol("axi4") == Protocol.AXI4
        assert Protocol("axi4_lite") == Protocol.AXI4L


# ═══════════════════════════════════════════════════════════════════
# §14  Full fixture-based parse — NPU 3D sub-kernel specs
# ═══════════════════════════════════════════════════════════════════


class TestFullSubKernelSpecParse:

    def test_fmapio_full(self, fmapio_spec_yaml):
        """fmapIO: 4 interfaces, DDR memory region, 9 parameters."""
        spec = parse_kernel_spec(str(fmapio_spec_yaml))
        assert spec.kernel_name == "fmapIO"
        assert set(spec.interface_names()) == {"ctrl", "ddr", "ifm_out", "ofm_in"}
        assert "ddr" in spec.memory_regions

    def test_bias_loader_full(self, bias_loader_spec_yaml):
        """bias_loader: 3 interfaces (ctrl, ddr, bias_out)."""
        spec = parse_kernel_spec(str(bias_loader_spec_yaml))
        assert spec.kernel_name == "bias_loader"
        assert set(spec.interface_names()) == {"ctrl", "ddr", "bias_out"}

    def test_weight_loader_full(self, weight_loader_spec_yaml):
        """weight_loader: 3 interfaces (ctrl, hbm, wgt_out)."""
        spec = parse_kernel_spec(str(weight_loader_spec_yaml))
        assert spec.kernel_name == "weight_loader"
        assert set(spec.interface_names()) == {"ctrl", "hbm", "wgt_out"}

    def test_mac_atu_full(self, mac_atu_spec_yaml):
        """mac_atu: AXI4-Lite only + 3 AXIS (no external AXI4)."""
        spec = parse_kernel_spec(str(mac_atu_spec_yaml))
        assert spec.kernel_name == "mac_atu"
        assert set(spec.interface_names()) == {"ctrl", "ifm_in", "wgt_in", "psum_out"}
        # No memory region (internal streams only)
        assert spec.memory_regions == {}
