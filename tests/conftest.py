"""Shared fixtures for vTen tests.

Generic fixtures live here; NPU_3D-specific fixtures are in fixtures/npu_3d.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Register NPU_3D fixtures as a plugin so pytest discovers them
pytest_plugins = ["tests.fixtures.npu_3d"]


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
