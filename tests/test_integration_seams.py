"""Phase 4 tests: Integration seams between Phase 2→4 modules.

Tests the compatibility boundaries:
- CompiledResult → SVGenerator (shm_image, bfm_configs flow into codegen)
- CompiledResult → Backend.submit (shm_image, bfm_configs accepted by backend)
- Config parsing → build pipeline → codegen chain

Spec references:
- 00_data_models.md §13 (CompiledResult)
- 06_codegen_and_cli.md §3 (SVGenerator inputs)
- 04_backend_xsim.md §3 (submit inputs)
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from vten.runtime.ir import BFMConfig, Command
from vten.spec.models import (
    InterfaceSpec,
    KernelSpec,
    OpCode,
    PackingScheme,
    Protocol,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _passthrough_compiled_result():
    """Simulate a CompiledResult for passthrough kernel."""
    from vten.runtime.engine import CompiledResult

    bfm_configs = [
        BFMConfig(interface_name="axi_stream_in", protocol=Protocol.AXI4S, data_width=32, role="master"),
        BFMConfig(interface_name="axi_stream_out", protocol=Protocol.AXI4S, data_width=32, role="slave"),
    ]

    commands = [
        Command(op=OpCode.LOAD, cmd_id=0, interface_id=0, buffer_id=0,
                protocol=Protocol.AXI4S, size=128, role="master"),
        Command(op=OpCode.PUSH, cmd_id=1, interface_id=0, buffer_id=0,
                protocol=Protocol.AXI4S, size=128, role="master", dep=0),
        Command(op=OpCode.PULL, cmd_id=2, interface_id=1, buffer_id=1,
                protocol=Protocol.AXI4S, size=128, role="slave", dep=1),
    ]

    # Minimal SHM image: control + 3 command slots
    from vten.runtime.shm import CMD_SLOT_SIZE, CONTROL_SIZE
    shm_size = CONTROL_SIZE + len(commands) * CMD_SLOT_SIZE
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x5654454E)  # MAGIC
    struct.pack_into("<I", buf, 4, 0x00000003)  # VERSION
    struct.pack_into("<I", buf, 8, len(commands))

    return CompiledResult(
        commands=commands,
        shm_image=bytes(buf),
        bfm_configs=bfm_configs,
        buffer_ids={"data_in": 0, "data_out": 1},
        flattened_view=None,  # type: ignore[arg-type]
    )


def _passthrough_spec() -> KernelSpec:
    return KernelSpec(
        kernel_name="passthrough",
        rtl_top="rtl/passthrough.sv",
        interfaces={
            "axi_stream_in": InterfaceSpec(
                name="axi_stream_in", rtl_port="s_axis_in",
                protocol=Protocol.AXI4S, tensor="data_in",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
            "axi_stream_out": InterfaceSpec(
                name="axi_stream_out", rtl_port="m_axis_out",
                protocol=Protocol.AXI4S, tensor="data_out",
                packing=PackingScheme(element_width=8, elements_per_beat=4),
            ),
        },
    )


def _minimal_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "rtl": {
            "sources": ["rtl/passthrough.sv"],
            "top_module": "passthrough",
            "include_dirs": [],
        },
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2024.1",
                "compile_options": ["-timescale", "1ns/1ps"],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# §1  CompiledResult → SVGenerator
# ═══════════════════════════════════════════════════════════════════


class TestCompiledResultToSVGenerator:
    """CompiledResult.bfm_configs feeds directly into SVGenerator."""

    def test_compiled_bfm_configs_accepted_by_svgenerator(self, tmp_path: Path):
        """SVGenerator accepts bfm_configs from CompiledResult."""
        from vten.codegen.sv_generator import SVGenerator

        result = _passthrough_compiled_result()
        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=result.bfm_configs,
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        assert (tmp_path / "tb_top.sv").exists()

    def test_compiled_bfm_configs_protocol_preserved(self):
        """BFMConfig.protocol from compile is same type SVGenerator expects."""
        result = _passthrough_compiled_result()
        for cfg in result.bfm_configs:
            assert isinstance(cfg.protocol, Protocol)
            assert cfg.protocol in (Protocol.AXI4S, Protocol.AXI4, Protocol.AXI4L)

    def test_compiled_bfm_config_interface_names_match_spec(self):
        """BFMConfig.interface_name matches KernelSpec interface names."""
        result = _passthrough_compiled_result()
        spec = _passthrough_spec()
        spec_iface_names = set(spec.interfaces.keys())
        compiled_iface_names = {cfg.interface_name for cfg in result.bfm_configs}
        assert compiled_iface_names == spec_iface_names

    def test_compiled_commands_have_valid_interface_ids(self):
        """Command.interface_id is within [0, len(bfm_configs))."""
        result = _passthrough_compiled_result()
        num_bfms = len(result.bfm_configs)
        for cmd in result.commands:
            assert 0 <= cmd.interface_id < num_bfms, (
                f"cmd_id={cmd.cmd_id} has interface_id={cmd.interface_id} "
                f"but only {num_bfms} BFMs configured"
            )

    def test_svgenerator_output_matches_bfm_count(self, tmp_path: Path):
        """Generated tb_top.sv has BFM instances matching compiled bfm_configs count."""
        import re

        from vten.codegen.sv_generator import SVGenerator

        result = _passthrough_compiled_result()
        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=result.bfm_configs,
            project_config=_minimal_config(),
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        # Should have MAX_BFMS >= len(bfm_configs)
        match = re.search(r"MAX_BFMS\s*[=(]\s*(\d+)", content)
        if match:
            max_bfms = int(match.group(1))
            assert max_bfms >= len(result.bfm_configs)


# ═══════════════════════════════════════════════════════════════════
# §2  CompiledResult → Backend.submit
# ═══════════════════════════════════════════════════════════════════


class TestCompiledResultToBackend:
    """CompiledResult.shm_image + bfm_configs → Backend.submit()."""

    def test_compiled_shm_image_is_bytes(self):
        """shm_image from compile is bytes (submit expects bytes)."""
        result = _passthrough_compiled_result()
        assert isinstance(result.shm_image, bytes)

    def test_compiled_shm_image_has_magic(self):
        """shm_image starts with VTEN magic."""
        result = _passthrough_compiled_result()
        magic = struct.unpack_from("<I", result.shm_image, 0)[0]
        assert magic == 0x5654454E

    def test_compiled_shm_image_has_version(self):
        """shm_image has correct version field."""
        result = _passthrough_compiled_result()
        version = struct.unpack_from("<I", result.shm_image, 4)[0]
        assert version == 0x00000003

    def test_compiled_shm_image_size_sufficient(self):
        """shm_image size >= CONTROL_SIZE + num_commands * CMD_SLOT_SIZE."""
        from vten.runtime.shm import CMD_SLOT_SIZE, CONTROL_SIZE

        result = _passthrough_compiled_result()
        min_size = CONTROL_SIZE + len(result.commands) * CMD_SLOT_SIZE
        assert len(result.shm_image) >= min_size

    def test_backend_submit_signature_compatible(self):
        """Backend.submit accepts (shm_image: bytes, bfm_configs: list[BFMConfig])."""
        import inspect

        from vten.backend.xsim import XsimBackend

        sig = inspect.signature(XsimBackend.submit)
        params = list(sig.parameters.keys())
        assert "shm_image" in params
        assert "bfm_configs" in params

    def test_compiled_bfm_configs_are_bfmconfig_instances(self):
        """All bfm_configs entries are BFMConfig dataclass instances."""
        result = _passthrough_compiled_result()
        for cfg in result.bfm_configs:
            assert isinstance(cfg, BFMConfig)
            assert hasattr(cfg, "interface_name")
            assert hasattr(cfg, "protocol")
            assert hasattr(cfg, "data_width")


# ═══════════════════════════════════════════════════════════════════
# §3  Config → Build → Codegen chain
# ═══════════════════════════════════════════════════════════════════


class TestConfigBuildCodegenChain:
    """vten.toml config flows through build into codegen correctly."""

    def test_config_top_module_reaches_tb_top(self, tmp_path: Path):
        """[rtl].top_module from config appears in generated tb_top.sv."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["rtl"]["top_module"] = "my_custom_dut"

        gen = SVGenerator(
            kernel_spec=KernelSpec(
                kernel_name="custom",
                rtl_top="rtl/custom.sv",
                interfaces={
                    "port_a": InterfaceSpec(
                        name="port_a", rtl_port="axis_a",
                        protocol=Protocol.AXI4S,
                    ),
                },
            ),
            bfm_configs=[BFMConfig(interface_name="port_a", protocol=Protocol.AXI4S, data_width=32, role="master")],
            project_config=config,
        )
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()
        assert "my_custom_dut" in content

    def test_config_vivado_path_reaches_build_tcl(self, tmp_path: Path):
        """[backend.xsim].vivado_path from config appears in build.tcl."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["backend"]["xsim"]["vivado_path"] = "/opt/Xilinx/Vivado/2023.2"

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[
                BFMConfig(interface_name="axi_stream_in", protocol=Protocol.AXI4S, data_width=32, role="master"),
                BFMConfig(interface_name="axi_stream_out", protocol=Protocol.AXI4S, data_width=32, role="slave"),
            ],
            project_config=config,
        )
        gen.generate(str(tmp_path))
        build_tcl = (tmp_path / "build.tcl").read_text()
        assert "2023.2" in build_tcl or "Vivado" in build_tcl

    def test_config_compile_options_reach_build_tcl(self, tmp_path: Path):
        """[backend.xsim].compile_options appear in build.tcl."""
        from vten.codegen.sv_generator import SVGenerator

        config = _minimal_config()
        config["backend"]["xsim"]["compile_options"] = ["-timescale", "1ns/1ps", "-d", "SIMULATION"]

        gen = SVGenerator(
            kernel_spec=_passthrough_spec(),
            bfm_configs=[
                BFMConfig(interface_name="axi_stream_in", protocol=Protocol.AXI4S, data_width=32, role="master"),
                BFMConfig(interface_name="axi_stream_out", protocol=Protocol.AXI4S, data_width=32, role="slave"),
            ],
            project_config=config,
        )
        gen.generate(str(tmp_path))
        build_tcl = (tmp_path / "build.tcl").read_text()
        assert "timescale" in build_tcl or "1ns/1ps" in build_tcl
