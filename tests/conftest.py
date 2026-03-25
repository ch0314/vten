"""Shared fixtures for vTen Phase 1 tests.

All fixtures reflect NPU 3D accelerator patterns (specs/npu_3d_analysis.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ── Helper ─────────────────────────────────────────────────────────


def _write_yaml(directory: Path, filename: str, data: dict) -> Path:
    p = directory / filename
    p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return p


# ═══════════════════════════════════════════════════════════════════
# AXI4-Stream — passthrough (가장 단순한 케이스)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def passthrough_spec_yaml(tmp_path: Path) -> Path:
    """Minimal AXI4-Stream passthrough kernel spec."""
    data = {
        "kernel": "passthrough",
        "rtl_top": "rtl/passthrough.sv",
        "parameters": {"SIZE": "${SIZE}"},
        "interfaces": {
            "axi_stream_in": {
                "rtl_port": "s_axis_in",
                "protocol": "axi4_stream",
                "tensor": "data_in",
                "packing": {
                    "element_width": 8,
                    "elements_per_beat": 4,
                },
            },
            "axi_stream_out": {
                "rtl_port": "m_axis_out",
                "protocol": "axi4_stream",
                "tensor": "data_out",
                "packing": {
                    "element_width": 8,
                    "elements_per_beat": 4,
                },
            },
        },
    }
    return _write_yaml(tmp_path, "passthrough.yaml", data)


# ═══════════════════════════════════════════════════════════════════
# NPU 3D sub-kernel specs — per-IP (npu_3d_analysis.md §4)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def fmapio_spec_yaml(tmp_path: Path) -> Path:
    """fmapIO sub-kernel: AXI4-Lite ctrl + AXI4 DDR + AXIS internal."""
    data = {
        "kernel": "fmapIO",
        "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
        "parameters": {
            "IN_DEPTH": "${IN_DEPTH}", "IN_HEIGHT": "${IN_HEIGHT}",
            "IN_WIDTH": "${IN_WIDTH}", "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}",
            "IFM_STRIDE": "${IFM_STRIDE}", "OFM_STRIDE": "${OFM_STRIDE}",
            "IS_CONCAT": "${IS_CONCAT}", "CONCAT_CH": "${CONCAT_CH}",
        },
        "memory_regions": {
            "ddr": {"base": 0x0000_0000, "size": 0x1_0000_0000, "alignment": 4096},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "registers": [
                    {"name": "in_depth", "offset": 0x014, "auto_bind": {"param": "${IN_DEPTH}"}},
                    {"name": "in_height", "offset": 0x018, "auto_bind": {"param": "${IN_HEIGHT}"}},
                    {"name": "in_width", "offset": 0x01C, "auto_bind": {"param": "${IN_WIDTH}"}},
                    {"name": "in_ch", "offset": 0x020, "auto_bind": {"param": "${IN_CH}"}},
                    {"name": "out_ch", "offset": 0x024, "auto_bind": {"param": "${OUT_CH}"}},
                    {"name": "ifm_stride", "offset": 0x028, "auto_bind": {"param": "${IFM_STRIDE}"}},
                    {"name": "ofm_stride", "offset": 0x02C, "auto_bind": {"param": "${OFM_STRIDE}"}},
                    {"name": "is_concat", "offset": 0x030, "auto_bind": {"param": "${IS_CONCAT}"}},
                    {"name": "concat_ch", "offset": 0x034, "auto_bind": {"param": "${CONCAT_CH}"}},
                    {"name": "ifm_addr_lsb", "offset": 0x038,
                     "auto_bind": {"tensor": "ifm", "value": "address", "bits": "31:0"}},
                    {"name": "ifm_addr_msb", "offset": 0x03C,
                     "auto_bind": {"tensor": "ifm", "value": "address", "bits": "63:32"}},
                    {"name": "ofm_addr_lsb", "offset": 0x040,
                     "auto_bind": {"tensor": "ofm", "value": "address", "bits": "31:0"}},
                    {"name": "ofm_addr_msb", "offset": 0x044,
                     "auto_bind": {"tensor": "ofm", "value": "address", "bits": "63:32"}},
                    {"name": "concat_addr_lsb", "offset": 0x048,
                     "auto_bind": {"tensor": "concat", "value": "address", "bits": "31:0"}},
                    {"name": "concat_addr_msb", "offset": 0x04C,
                     "auto_bind": {"tensor": "concat", "value": "address", "bits": "63:32"}},
                    {"name": "vsync", "offset": 0x050, "fields": {"trigger": "0:0"}},
                    {"name": "layer_done", "offset": 0x054, "fields": {"done": "0:0"}},
                ],
            },
            "ddr": {
                "rtl_port": "m_axi_ddr",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "ddr",
                "tensors": ["ifm", "ofm", "concat"],
                "packing": {
                    "element_width": 8,
                    "elements_per_beat": 32,
                    "alignment": "packed",
                },
            },
            "ifm_out": {
                "rtl_port": "m_axis_ifm",
                "protocol": "axi4_stream",
                "tensor": "ifm",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
            "ofm_in": {
                "rtl_port": "s_axis_ofm",
                "protocol": "axi4_stream",
                "tensor": "ofm",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
        },
    }
    return _write_yaml(tmp_path, "fmapIO.yaml", data)


@pytest.fixture()
def bias_loader_spec_yaml(tmp_path: Path) -> Path:
    """bias_loader sub-kernel: 4 registers + AXI4 DDR (int32 bias)."""
    data = {
        "kernel": "bias_loader",
        "rtl_top": "design/bias_loader/rtl/bias_loader_top.sv",
        "parameters": {"OUT_CH": "${OUT_CH}"},
        "memory_regions": {
            "ddr": {"base": 0x0000_0000, "size": 0x1_0000_0000},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "registers": [
                    {"name": "bias_addr_lsb", "offset": 0x010,
                     "auto_bind": {"tensor": "bias", "value": "address", "bits": "31:0"}},
                    {"name": "bias_addr_msb", "offset": 0x014,
                     "auto_bind": {"tensor": "bias", "value": "address", "bits": "63:32"}},
                    {"name": "out_ch", "offset": 0x018,
                     "auto_bind": {"param": "${OUT_CH}"}},
                    {"name": "vsync", "offset": 0x01C, "fields": {"trigger": "0:0"}},
                ],
            },
            "ddr": {
                "rtl_port": "m_axi_ddr",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "ddr",
                "tensor": "bias",
                "packing": {
                    "element_width": 32,
                    "elements_per_beat": 8,
                },
            },
            "bias_out": {
                "rtl_port": "m_axis_bias",
                "protocol": "axi4_stream",
                "tensor": "bias",
                "packing": {"element_width": 32, "elements_per_beat": 8},
            },
        },
    }
    return _write_yaml(tmp_path, "bias_loader.yaml", data)


@pytest.fixture()
def weight_loader_spec_yaml(tmp_path: Path) -> Path:
    """weight_loader: 71 registers + HBM 32-port split."""
    regs = [
        {"name": "in_width", "offset": 0x010, "auto_bind": {"param": "${IN_WIDTH}"}},
        {"name": "in_height", "offset": 0x014, "auto_bind": {"param": "${IN_HEIGHT}"}},
        {"name": "in_depth", "offset": 0x018, "auto_bind": {"param": "${IN_DEPTH}"}},
    ]
    for i in range(32):
        regs.append({"name": f"wgt_addr_{i:02d}_lsb", "offset": 0x01C + 8 * i})
        regs.append({"name": f"wgt_addr_{i:02d}_msb", "offset": 0x020 + 8 * i})
    regs.extend([
        {"name": "vsync", "offset": 0x11C, "fields": {"trigger": "0:0"}},
        {"name": "kernel_size", "offset": 0x120, "auto_bind": {"param": "${KERNEL_SIZE}"}},
        {"name": "in_ch", "offset": 0x124, "auto_bind": {"param": "${IN_CH}"}},
        {"name": "out_ch", "offset": 0x128, "auto_bind": {"param": "${OUT_CH}"}},
    ])

    hbm_ports = [{"name": f"hbm_m{i:02d}_axi", "base_addr": 0} for i in range(32)]

    data = {
        "kernel": "weight_loader",
        "rtl_top": "design/weight_loader/rtl/weight_loader_top.sv",
        "parameters": {
            "IN_WIDTH": "${IN_WIDTH}", "IN_HEIGHT": "${IN_HEIGHT}", "IN_DEPTH": "${IN_DEPTH}",
            "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}", "KERNEL_SIZE": "${KERNEL_SIZE}",
        },
        "memory_regions": {
            "hbm": {"base": 0x0000_0000, "size": 0x1_0000_0000},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "registers": regs,
            },
            "hbm": {
                "rtl_port": "m_axi_hbm",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "hbm",
                "tensor": "weight",
                "packing": {"element_width": 8, "elements_per_beat": 32},
                "split": {"mode": "channel_interleave", "ports": hbm_ports},
            },
            "wgt_out": {
                "rtl_port": "m_axis_wgt",
                "protocol": "axi4_stream",
                "tensor": "weight",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
        },
    }
    return _write_yaml(tmp_path, "weight_loader.yaml", data)


@pytest.fixture()
def mac_atu_spec_yaml(tmp_path: Path) -> Path:
    """mac_atu: AXI4-Lite only (internal AXIS, no external AXI4)."""
    data = {
        "kernel": "mac_atu",
        "rtl_top": "design/mac/rtl/mac_atu_top_wrapper.sv",
        "parameters": {
            "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}",
            "IN_WIDTH": "${IN_WIDTH}", "IN_HEIGHT": "${IN_HEIGHT}",
            "IS_CONCAT": "${IS_CONCAT}", "CONCAT_CH": "${CONCAT_CH}",
            "IFM_DTYPE": "${IFM_DTYPE}",
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "registers": [
                    {"name": "ifm_is_signed", "offset": 0x010, "auto_bind": {"param": "${IFM_DTYPE}"}},
                    {"name": "in_ch", "offset": 0x014, "auto_bind": {"param": "${IN_CH}"}},
                    {"name": "out_ch", "offset": 0x018, "auto_bind": {"param": "${OUT_CH}"}},
                    {"name": "is_concat", "offset": 0x01C, "auto_bind": {"param": "${IS_CONCAT}"}},
                    {"name": "concat_is_signed", "offset": 0x020, "auto_bind": {"param": "${IFM_DTYPE}"}},
                    {"name": "concat_ch", "offset": 0x024, "auto_bind": {"param": "${CONCAT_CH}"}},
                    {"name": "in_width", "offset": 0x028, "auto_bind": {"param": "${IN_WIDTH}"}},
                    {"name": "in_height", "offset": 0x02C, "auto_bind": {"param": "${IN_HEIGHT}"}},
                ],
            },
            "ifm_in": {
                "rtl_port": "s_axis_ifm",
                "protocol": "axi4_stream",
                "tensor": "ifm",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
            "wgt_in": {
                "rtl_port": "s_axis_wgt",
                "protocol": "axi4_stream",
                "tensor": "weight",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
            "psum_out": {
                "rtl_port": "m_axis_psum",
                "protocol": "axi4_stream",
                "tensor": "psum",
                "packing": {"element_width": 32, "elements_per_beat": 8},
            },
        },
    }
    return _write_yaml(tmp_path, "mac_atu.yaml", data)


# ═══════════════════════════════════════════════════════════════════
# Conv3D Composite — full NPU 3D with register_banks + auto_bind
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def conv3d_composite_spec_yamls(tmp_path: Path) -> dict[str, Path]:
    """Conv3D composite kernel: 4 sub-kernels sharing AXI4-Lite via register_banks.

    Returns a dict of {kernel_name: yaml_path} for each sub-kernel spec,
    plus a top-level composite spec under 'npu_top' key.
    """
    specs_dir = tmp_path / "kernels"
    specs_dir.mkdir()

    # ── fmapio sub-kernel ──
    fmapio = {
        "kernel": "fmapIO",
        "rtl_top": "design/fmapIO/rtl/fmapIO_top.sv",
        "clock": {"name": "ap_clk"},
        "reset": {"name": "ap_aresetn", "active_low": True},
        "parameters": {
            "IN_DEPTH": "${IN_DEPTH}", "IN_HEIGHT": "${IN_HEIGHT}",
            "IN_WIDTH": "${IN_WIDTH}", "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}",
        },
        "memory_regions": {
            "ddr": {"base": 0x0000_0000, "size": 0x1_0000_0000, "alignment": 4096},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "register_banks": {
                    "fmapio": {"base_offset": 0x000},
                },
                "registers": [
                    {"name": "in_depth", "offset": 0x014, "auto_bind": {"param": "${IN_DEPTH}"}},
                    {"name": "ifm_addr_lsb", "offset": 0x038,
                     "auto_bind": {"tensor": "ifm", "value": "address", "bits": "31:0"}},
                    {"name": "ifm_addr_msb", "offset": 0x03C,
                     "auto_bind": {"tensor": "ifm", "value": "address", "bits": "63:32"}},
                    {"name": "vsync", "offset": 0x050, "fields": {"trigger": "0:0"}},
                    {"name": "layer_done", "offset": 0x054, "fields": {"done": "0:0"}},
                ],
            },
            "ddr": {
                "rtl_port": "m_axi_ddr",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "ddr",
                "tensor": "ifm",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
        },
    }

    # ── weight_loader sub-kernel (with HBM split) ──
    wgt_regs = [
        {"name": "in_width", "offset": 0x010, "auto_bind": {"param": "${IN_WIDTH}"}},
    ]
    for i in range(4):  # Simplified: 4 HBM ports instead of 32
        wgt_regs.append({"name": f"wgt_addr_{i:02d}_lsb", "offset": 0x01C + 8 * i})
        wgt_regs.append({"name": f"wgt_addr_{i:02d}_msb", "offset": 0x020 + 8 * i})
    wgt_regs.append({"name": "vsync", "offset": 0x05C, "fields": {"trigger": "0:0"}})

    hbm_ports = [{"name": f"hbm_m{i:02d}_axi", "base_addr": 0} for i in range(4)]

    weight_loader = {
        "kernel": "weight_loader",
        "rtl_top": "design/weight_loader/rtl/weight_loader_top.sv",
        "clock": {"name": "ap_clk"},
        "reset": {"name": "ap_aresetn", "active_low": True},
        "parameters": {"IN_WIDTH": "${IN_WIDTH}", "IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}"},
        "memory_regions": {
            "hbm": {"base": 0x0000_0000, "size": 0x1_0000_0000},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "register_banks": {
                    "weight_loader": {"base_offset": 0x1000},
                },
                "registers": wgt_regs,
            },
            "hbm": {
                "rtl_port": "m_axi_hbm",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "hbm",
                "tensor": "weight",
                "packing": {"element_width": 8, "elements_per_beat": 32},
                "split": {"mode": "channel_interleave", "ports": hbm_ports},
            },
        },
    }

    # ── mac_atu sub-kernel ──
    mac_atu = {
        "kernel": "mac_atu",
        "rtl_top": "design/mac/rtl/mac_atu_top_wrapper.sv",
        "clock": {"name": "ap_clk"},
        "reset": {"name": "ap_aresetn", "active_low": True},
        "parameters": {"IN_CH": "${IN_CH}", "OUT_CH": "${OUT_CH}"},
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "register_banks": {
                    "mac": {"base_offset": 0x2000},
                },
                "registers": [
                    {"name": "in_ch", "offset": 0x014, "auto_bind": {"param": "${IN_CH}"}},
                    {"name": "out_ch", "offset": 0x018, "auto_bind": {"param": "${OUT_CH}"}},
                ],
            },
            "ifm_in": {
                "rtl_port": "s_axis_ifm",
                "protocol": "axi4_stream",
                "tensor": "ifm",
                "packing": {"element_width": 8, "elements_per_beat": 32},
            },
            "psum_out": {
                "rtl_port": "m_axis_psum",
                "protocol": "axi4_stream",
                "tensor": "psum",
                "packing": {"element_width": 32, "elements_per_beat": 8},
            },
        },
    }

    # ── bias_loader sub-kernel ──
    bias_loader = {
        "kernel": "bias_loader",
        "rtl_top": "design/bias_loader/rtl/bias_loader_top.sv",
        "clock": {"name": "ap_clk"},
        "reset": {"name": "ap_aresetn", "active_low": True},
        "parameters": {"OUT_CH": "${OUT_CH}"},
        "memory_regions": {
            "ddr": {"base": 0x0000_0000, "size": 0x1_0000_0000},
        },
        "interfaces": {
            "ctrl": {
                "rtl_port": "s_axilite_ctrl",
                "protocol": "axi4_lite",
                "addr_width": 16,
                "register_banks": {
                    "bias": {"base_offset": 0x3000},
                },
                "registers": [
                    {"name": "bias_addr_lsb", "offset": 0x010,
                     "auto_bind": {"tensor": "bias", "value": "address", "bits": "31:0"}},
                    {"name": "bias_addr_msb", "offset": 0x014,
                     "auto_bind": {"tensor": "bias", "value": "address", "bits": "63:32"}},
                    {"name": "out_ch", "offset": 0x018, "auto_bind": {"param": "${OUT_CH}"}},
                ],
            },
            "ddr": {
                "rtl_port": "m_axi_ddr",
                "protocol": "axi4",
                "data_width": 256,
                "addr_width": 64,
                "memory_region": "ddr",
                "tensor": "bias",
                "packing": {"element_width": 32, "elements_per_beat": 8},
            },
        },
    }

    results = {}
    for name, data in [
        ("fmapio", fmapio),
        ("weight_loader", weight_loader),
        ("mac_atu", mac_atu),
        ("bias_loader", bias_loader),
    ]:
        kdir = specs_dir / name
        kdir.mkdir()
        results[name] = _write_yaml(kdir, "kernel_spec.yaml", data)

    return results
