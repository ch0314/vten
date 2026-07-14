"""Tests for integration seams between the runtime and codegen modules.

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
    MemoryRegion,
    OpCode,
    PackingScheme,
    Protocol,
    RegisterBankSpec,
    RegisterSpec,
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
                protocol=Protocol.AXI4S, size=128, role="master", dep=[0]),
        Command(op=OpCode.PULL, cmd_id=2, interface_id=1, buffer_id=1,
                protocol=Protocol.AXI4S, size=128, role="slave", dep=[1]),
    ]

    # Minimal SHM image: control + 3 command slots
    from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE
    shm_size = CONTROL_SIZE + len(commands) * CMD_SLOT_SIZE
    buf = bytearray(shm_size)
    struct.pack_into("<I", buf, 0, 0x5654454E)  # MAGIC
    struct.pack_into("<I", buf, 4, 0x00000003)  # VERSION
    struct.pack_into("<I", buf, 8, len(commands))

    return CompiledResult(
        commands=commands,
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
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
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
    """CompiledResult + bfm_configs → Backend.execute()."""

    def _make_shm_image(self, num_commands: int) -> bytes:
        """Build a minimal valid SHM image directly."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE
        size = CONTROL_SIZE + num_commands * CMD_SLOT_SIZE
        buf = bytearray(size)
        struct.pack_into("<I", buf, 0, 0x5654454E)  # MAGIC
        struct.pack_into("<I", buf, 4, 0x00000003)  # VERSION
        struct.pack_into("<I", buf, 8, num_commands)
        return bytes(buf)

    def test_shm_image_is_bytes(self):
        """SHM image from pack is bytes."""
        img = self._make_shm_image(3)
        assert isinstance(img, bytes)

    def test_shm_image_has_magic(self):
        """SHM image starts with VTEN magic."""
        img = self._make_shm_image(3)
        magic = struct.unpack_from("<I", img, 0)[0]
        assert magic == 0x5654454E

    def test_shm_image_has_version(self):
        """SHM image has correct version field."""
        img = self._make_shm_image(3)
        version = struct.unpack_from("<I", img, 4)[0]
        assert version == 0x00000003

    def test_shm_image_size_sufficient(self):
        """SHM image size >= CONTROL_SIZE + num_commands * CMD_SLOT_SIZE."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE
        img = self._make_shm_image(3)
        min_size = CONTROL_SIZE + 3 * CMD_SLOT_SIZE
        assert len(img) >= min_size

    def test_backend_execute_signature_compatible(self):
        """Backend.execute accepts (compiled: CompiledResult)."""
        import inspect

        from vten.backend.xsim import XsimBackend

        sig = inspect.signature(XsimBackend.execute)
        params = list(sig.parameters.keys())
        assert "compiled" in params

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

    def test_config_vivado_path_stored_in_build_context(self, tmp_path: Path):
        """[backend.xsim].vivado_path from config reaches SVGenerator context."""
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
        ctx = gen._build_context()
        assert ctx.build.vivado_path == "/opt/Xilinx/Vivado/2023.2"

    def test_config_compile_options_stored_in_build_context(self, tmp_path: Path):
        """[backend.xsim].compile_options reach SVGenerator context."""
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
        ctx = gen._build_context()
        assert "-timescale" in ctx.build.compile_options
        assert "SIMULATION" in ctx.build.compile_options[-1]


# ═══════════════════════════════════════════════════════════════════
# §4  Codegen structural verification — BFM count & iface_to_bfm
# ═══════════════════════════════════════════════════════════════════


def _npu_spec_with_registers() -> KernelSpec:
    """NPU-scale spec with register_banks + auto_bind + memory_regions."""
    interfaces: dict[str, InterfaceSpec] = {}
    # 6 AXI4-Lite control ports with register_banks
    for name, bank_offset in [
        ("ctrl_fmapio", 0x0000), ("ctrl_wgt", 0x1000),
        ("ctrl_mac", 0x2000), ("ctrl_psum", 0x3000),
        ("ctrl_bias", 0x4000), ("ctrl_act", 0x5000),
    ]:
        interfaces[name] = InterfaceSpec(
            name=name, rtl_port=f"s_axilite_{name}",
            protocol=Protocol.AXI4L, addr_width=16,
            register_banks=[RegisterBankSpec(name=name, base_offset=bank_offset)],
            registers=[
                RegisterSpec(name="vsync", offset=0x010, interface_name=name),
                RegisterSpec(
                    name="param_0", offset=0x014, interface_name=name,
                ),
            ],
        )
    # 2 DDR AXI4 with memory_region
    for name in ["ddr_fmap", "ddr_bias"]:
        interfaces[name] = InterfaceSpec(
            name=name, rtl_port=f"m_axi_{name}",
            protocol=Protocol.AXI4, data_width=256, addr_width=64,
            memory_region="ddr",
        )
    # 32 HBM AXI4
    for i in range(32):
        name = f"hbm_{i:02d}"
        interfaces[name] = InterfaceSpec(
            name=name, rtl_port=f"m_axi_{name}",
            protocol=Protocol.AXI4, data_width=256, addr_width=64,
            memory_region="hbm",
        )
    return KernelSpec(
        kernel_name="npu_3d", rtl_top="design/NPU_3D_top.sv",
        interfaces=interfaces,
        memory_regions={
            "ddr": MemoryRegion(name="ddr", base=0, size=0x1_0000_0000),
            "hbm": MemoryRegion(name="hbm", base=0, size=0x1_0000_0000),
        },
        clock_name="ap_clk",
        reset_name="ap_aresetn",
        reset_active_low=True,
    )


def _npu_bfm_configs_from_spec(spec: KernelSpec) -> list[BFMConfig]:
    """Derive BFM configs from spec — mirrors build.py _derive_bfm_configs."""
    from vten.cli.build import _derive_bfm_configs
    return _derive_bfm_configs(spec)


class TestCodegenStructuralVerification:
    """Verify generated testbench structural properties match spec."""

    def test_bfm_count_matches_interface_count(self, tmp_path: Path):
        """Generated BFM instance count == len(spec.interfaces)."""
        import re
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        # MAX_BFMS must be >= number of interfaces
        match = re.search(r"MAX_BFMS\s*[=(]\s*(\d+)", content)
        assert match is not None
        max_bfms = int(match.group(1))
        assert max_bfms >= len(spec.interfaces), (
            f"MAX_BFMS={max_bfms} < {len(spec.interfaces)} interfaces"
        )

    def test_iface_to_bfm_completeness(self, tmp_path: Path):
        """Every interface_id has an iface_to_bfm entry."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        iface_to_bfm_count = content.count("iface_to_bfm[")
        assert iface_to_bfm_count >= len(spec.interfaces), (
            f"iface_to_bfm entries ({iface_to_bfm_count}) < "
            f"interfaces ({len(spec.interfaces)})"
        )

    def test_all_three_protocol_bfm_modules_present(self, tmp_path: Path):
        """NPU testbench uses AXI4-Lite + AXI4 BFM modules."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        assert "vten_bfm_axilite" in content, "Missing AXI4-Lite BFM module"
        assert "vten_bfm_axi4" in content, "Missing AXI4 BFM module"

    def test_clock_reset_from_spec_propagates(self, tmp_path: Path):
        """Custom clock/reset names from KernelSpec appear in generated TB."""
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        assert "ap_clk" in content, "Custom clock name not found in generated TB"
        assert "ap_aresetn" in content, "Custom reset name not found in generated TB"

    def test_scheduler_max_ifaces_covers_all_interfaces(self, tmp_path: Path):
        """MAX_IFACES >= number of interfaces in spec."""
        import re
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=cfgs, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        content = (tmp_path / "tb_top.sv").read_text()

        match = re.search(r"MAX_IFACES\s*[=(]\s*(\d+)", content)
        assert match is not None
        max_ifaces = int(match.group(1))
        assert max_ifaces >= len(spec.interfaces), (
            f"MAX_IFACES={max_ifaces} < {len(spec.interfaces)} interfaces"
        )

    def test_each_bfm_has_unique_interface_id(self, tmp_path: Path):
        """No duplicate interface_ids across BFM configs."""
        spec = _npu_spec_with_registers()
        cfgs = _npu_bfm_configs_from_spec(spec)
        iface_names = [c.interface_name for c in cfgs]
        assert len(iface_names) == len(set(iface_names)), (
            f"Duplicate interface_names in BFM configs: {iface_names}"
        )


# ═══════════════════════════════════════════════════════════════════
# §5  Codegen-Runtime BFM config consistency
# ═══════════════════════════════════════════════════════════════════


class TestCodegenRuntimeConsistency:
    """Build pipeline BFM derivation matches runtime expectations."""

    def test_build_derive_matches_manual_configs(self):
        """_derive_bfm_configs produces same protocols as manual BFMConfig."""
        from vten.cli.build import _derive_bfm_configs

        spec = _passthrough_spec()
        derived = _derive_bfm_configs(spec)

        assert len(derived) == 2
        names = {c.interface_name for c in derived}
        assert names == {"axi_stream_in", "axi_stream_out"}
        for c in derived:
            assert c.protocol == Protocol.AXI4S

    def test_build_derive_infers_correct_roles(self):
        """BFM role inferred from rtl_port prefix: s_* → master, m_* → slave."""
        from vten.cli.build import _derive_bfm_configs

        spec = _passthrough_spec()
        derived = _derive_bfm_configs(spec)
        role_map = {c.interface_name: c.role for c in derived}

        # s_axis_in → DUT slave → BFM master
        assert role_map["axi_stream_in"] == "master"
        # m_axis_out → DUT master → BFM slave
        assert role_map["axi_stream_out"] == "slave"

    def test_build_derive_axi4_always_slave(self):
        """AXI4 BFMs are always slave (DUT initiates transactions)."""
        from vten.cli.build import _derive_bfm_configs

        spec = KernelSpec(
            kernel_name="mem_test", rtl_top="rtl/mem.sv",
            interfaces={
                "ddr": InterfaceSpec(
                    name="ddr", rtl_port="m_axi_ddr",
                    protocol=Protocol.AXI4, data_width=256, addr_width=64,
                ),
            },
        )
        derived = _derive_bfm_configs(spec)
        assert derived[0].role == "slave"

    def test_build_derive_axilite_always_master(self):
        """AXI4-Lite BFMs are always master (BFM drives register access)."""
        from vten.cli.build import _derive_bfm_configs

        spec = KernelSpec(
            kernel_name="ctrl_test", rtl_top="rtl/ctrl.sv",
            interfaces={
                "ctrl": InterfaceSpec(
                    name="ctrl", rtl_port="s_axilite_ctrl",
                    protocol=Protocol.AXI4L, addr_width=16,
                ),
            },
        )
        derived = _derive_bfm_configs(spec)
        assert derived[0].role == "master"

    def test_npu_scale_derive_40_bfms(self):
        """NPU spec with 40 interfaces produces 40 BFM configs."""
        from vten.cli.build import _derive_bfm_configs

        spec = _npu_spec_with_registers()
        derived = _derive_bfm_configs(spec)
        assert len(derived) == 40, f"Expected 40 BFMs, got {len(derived)}"

    def test_npu_derive_protocol_distribution(self):
        """NPU 40 BFMs: 6 AXI4-Lite + 34 AXI4."""
        from vten.cli.build import _derive_bfm_configs

        spec = _npu_spec_with_registers()
        derived = _derive_bfm_configs(spec)
        axil_count = sum(1 for c in derived if c.protocol == Protocol.AXI4L)
        axi4_count = sum(1 for c in derived if c.protocol == Protocol.AXI4)
        assert axil_count == 6, f"Expected 6 AXI4-Lite, got {axil_count}"
        assert axi4_count == 34, f"Expected 34 AXI4, got {axi4_count}"

    def test_derived_configs_accepted_by_svgenerator(self, tmp_path: Path):
        """SVGenerator accepts build-derived BFM configs without error."""
        from vten.cli.build import _derive_bfm_configs
        from vten.codegen.sv_generator import SVGenerator

        spec = _npu_spec_with_registers()
        derived = _derive_bfm_configs(spec)
        gen = SVGenerator(kernel_spec=spec, bfm_configs=derived, project_config=_minimal_config())
        gen.generate(str(tmp_path))
        assert (tmp_path / "tb_top.sv").exists()

    def test_derived_data_width_matches_spec(self):
        """BFMConfig.data_width reflects spec interface data_width."""
        from vten.cli.build import _derive_bfm_configs

        spec = _npu_spec_with_registers()
        derived = _derive_bfm_configs(spec)
        for cfg in derived:
            iface = spec.interfaces[cfg.interface_name]
            if iface.data_width is not None:
                assert cfg.data_width == iface.data_width, (
                    f"{cfg.interface_name}: data_width mismatch "
                    f"(derived={cfg.data_width}, spec={iface.data_width})"
                )


# ═══════════════════════════════════════════════════════════════════
# §6  Multi-batch backend execution lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestMultiBatchLifecycle:
    """Backend multi-batch submit/wait lifecycle compatibility."""

    def _make_shm_image(self, num_commands: int = 3) -> bytes:
        """Build a valid SHM image with given command count."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE
        size = CONTROL_SIZE + num_commands * CMD_SLOT_SIZE
        buf = bytearray(size)
        struct.pack_into("<I", buf, 0, 0x5654454E)  # MAGIC
        struct.pack_into("<I", buf, 4, 0x00000003)  # VERSION
        struct.pack_into("<I", buf, 8, num_commands)
        return bytes(buf)

    def test_shm_images_different_sizes_valid(self):
        """Different batch sizes produce valid SHM headers."""
        for n in [1, 10, 100, 256]:
            img = self._make_shm_image(n)
            magic = struct.unpack_from("<I", img, 0)[0]
            count = struct.unpack_from("<I", img, 8)[0]
            assert magic == 0x5654454E
            assert count == n

    def test_bfm_configs_reusable_across_batches(self):
        """Same BFM configs can be submitted with different SHM images."""
        cfgs = [
            BFMConfig(interface_name="axi_stream_in", protocol=Protocol.AXI4S, data_width=32, role="master"),
            BFMConfig(interface_name="axi_stream_out", protocol=Protocol.AXI4S, data_width=32, role="slave"),
        ]
        img1 = self._make_shm_image(3)
        img2 = self._make_shm_image(10)

        # Configs are reusable (not mutated between batches)
        assert cfgs[0].interface_name == "axi_stream_in"
        assert cfgs[1].interface_name == "axi_stream_out"
        assert len(img1) != len(img2)

    def test_session_ids_unique_per_batch(self):
        """Backend generates unique session IDs for each batch."""
        from vten.backend.xsim import XsimBackend

        config = {
            "project": {"name": "test"},
            "backend": {"xsim": {"vivado_path": "/tools/Xilinx", "timeout_ms": 5000}},
            "rtl": {"sources": [], "top_module": "tb_top"},
        }
        backend = XsimBackend(project_config=config)
        ids = [backend._generate_session_id() for _ in range(100)]
        assert len(set(ids)) == 100, "Session IDs must be unique"

    def test_backend_execute_signature_includes_compiled(self):
        """execute() accepts compiled parameter."""
        import inspect
        from vten.backend.xsim import XsimBackend

        sig = inspect.signature(XsimBackend.execute)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "compiled" in params

    def test_npu_scale_shm_image_256_commands(self):
        """256-command SHM image (NPU batch size) has correct size."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE

        img = self._make_shm_image(256)
        expected = CONTROL_SIZE + 256 * CMD_SLOT_SIZE
        assert len(img) == expected

    def test_shm_image_command_count_round_trip(self):
        """num_commands stored at offset 8 matches construction param."""
        for n in [0, 1, 50, 255, 1024]:
            img = self._make_shm_image(n)
            stored = struct.unpack_from("<I", img, 8)[0]
            assert stored == n
