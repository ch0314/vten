"""SVGenerator: Jinja2-based SystemVerilog testbench code generation.

Spec reference: 06_codegen_and_cli.md §1-3
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from vten.runtime.ir import BFMConfig
from vten.spec.models import KernelSpec, Protocol


# ── Template Context Dataclasses — 06_codegen_and_cli.md §2 ──


@dataclass
class DUTPort:
    name: str
    direction: str  # "input" | "output" | "inout"
    width: int
    connected_to: str


@dataclass
class BFMInstance:
    name: str
    module_name: str
    protocol: str  # "axi4_stream" | "axi4" | "axi4_lite"
    data_width: int
    role: str  # "master" | "slave"
    rtl_port_prefix: str
    parameters: dict
    interface_id: int


@dataclass
class TestbenchContext:
    project_name: str
    top_module: str
    session_id: str
    dut_ports: list[DUTPort]
    bfms: list[BFMInstance]
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True
    clock_period_ns: float = 10.0
    timeout_cycles: int = 1_000_000


@dataclass
class BuildContext:
    vivado_path: str
    rtl_sources: list[str]
    include_dirs: list[str]
    generated_sv: list[str]
    vten_sv_dir: str
    dpi_c_source: str
    compile_options: list[str]
    timescale: str = "1ns/1ps"
    top_module: str = "tb_top"


@dataclass
class GeneratorContext:
    tb: TestbenchContext
    build: BuildContext


# ── Protocol → BFM module mapping ──

_PROTOCOL_MODULE_MAP = {
    Protocol.AXI4S: "vten_bfm_axi4s",
    Protocol.AXI4: "vten_bfm_axi4",
    Protocol.AXI4L: "vten_bfm_axilite",
}


# ── SVGenerator ──


class SVGenerator:
    """Generates SystemVerilog testbench and build scripts from Jinja2 templates."""

    def __init__(
        self,
        kernel_spec: KernelSpec,
        bfm_configs: list[BFMConfig],
        project_config: dict,
    ) -> None:
        self.spec = kernel_spec
        self.bfm_configs = bfm_configs
        self.config = project_config

    def _module_for_protocol(self, protocol: Protocol) -> str:
        return _PROTOCOL_MODULE_MAP[protocol]

    def _derive_dut_ports(self, bfms: list[BFMInstance]) -> list[DUTPort]:
        """Derive DUT port list from BFM signal topology."""
        ports: list[DUTPort] = []
        for bfm in bfms:
            prefix = bfm.rtl_port_prefix
            if bfm.protocol == "axi4_stream":
                signals = [
                    (f"{prefix}_tdata", "input" if bfm.role == "slave" else "output", bfm.data_width),
                    (f"{prefix}_tvalid", "input" if bfm.role == "slave" else "output", 1),
                    (f"{prefix}_tready", "output" if bfm.role == "slave" else "input", 1),
                    (f"{prefix}_tlast", "input" if bfm.role == "slave" else "output", 1),
                ]
            elif bfm.protocol == "axi4":
                signals = [
                    (f"{prefix}_araddr", "input", 64),
                    (f"{prefix}_arlen", "input", 8),
                    (f"{prefix}_arsize", "input", 3),
                    (f"{prefix}_arburst", "input", 2),
                    (f"{prefix}_arvalid", "input", 1),
                    (f"{prefix}_arready", "output", 1),
                    (f"{prefix}_rdata", "output", bfm.data_width),
                    (f"{prefix}_rresp", "output", 2),
                    (f"{prefix}_rlast", "output", 1),
                    (f"{prefix}_rvalid", "output", 1),
                    (f"{prefix}_rready", "input", 1),
                    (f"{prefix}_awaddr", "input", 64),
                    (f"{prefix}_awlen", "input", 8),
                    (f"{prefix}_awsize", "input", 3),
                    (f"{prefix}_awburst", "input", 2),
                    (f"{prefix}_awvalid", "input", 1),
                    (f"{prefix}_awready", "output", 1),
                    (f"{prefix}_wdata", "input", bfm.data_width),
                    (f"{prefix}_wstrb", "input", bfm.data_width // 8),
                    (f"{prefix}_wlast", "input", 1),
                    (f"{prefix}_wvalid", "input", 1),
                    (f"{prefix}_wready", "output", 1),
                    (f"{prefix}_bresp", "output", 2),
                    (f"{prefix}_bvalid", "output", 1),
                    (f"{prefix}_bready", "input", 1),
                ]
            elif bfm.protocol == "axi4_lite":
                signals = [
                    (f"{prefix}_awaddr", "input", 32),
                    (f"{prefix}_awvalid", "input", 1),
                    (f"{prefix}_awready", "output", 1),
                    (f"{prefix}_wdata", "input", 32),
                    (f"{prefix}_wvalid", "input", 1),
                    (f"{prefix}_wready", "output", 1),
                    (f"{prefix}_araddr", "input", 32),
                    (f"{prefix}_arvalid", "input", 1),
                    (f"{prefix}_arready", "output", 1),
                    (f"{prefix}_rdata", "output", 32),
                    (f"{prefix}_rvalid", "output", 1),
                    (f"{prefix}_rready", "input", 1),
                ]
            else:
                signals = []

            for name, direction, width in signals:
                ports.append(DUTPort(
                    name=name,
                    direction=direction,
                    width=width,
                    connected_to=name,
                ))
        return ports

    def _build_context(self) -> GeneratorContext:
        """Build template context from spec and BFM configs."""
        bfms: list[BFMInstance] = []
        for i, cfg in enumerate(self.bfm_configs):
            iface = self.spec.get_interface(cfg.interface_name)
            params: dict = {"DATA_W": cfg.data_width}
            if cfg.protocol == Protocol.AXI4:
                params["ADDR_W"] = cfg.addr_width

            bfms.append(BFMInstance(
                name=f"bfm_{cfg.interface_name}",
                module_name=self._module_for_protocol(cfg.protocol),
                protocol=cfg.protocol.value,
                data_width=cfg.data_width,
                role=cfg.role,
                rtl_port_prefix=iface.rtl_port,
                parameters=params,
                interface_id=i,
            ))

        rtl_cfg = self.config.get("rtl", {})
        top_module = rtl_cfg.get("top_module", self.spec.kernel_name)

        # Derive DUT ports from BFM signal topology
        dut_ports = self._derive_dut_ports(bfms)

        tb = TestbenchContext(
            project_name=self.config.get("project", {}).get("name", ""),
            top_module=top_module,
            session_id=uuid.uuid4().hex[:16],
            dut_ports=dut_ports,
            bfms=bfms,
        )

        xsim_cfg = self.config.get("backend", {}).get("xsim", {})
        build = BuildContext(
            vivado_path=xsim_cfg.get("vivado_path", ""),
            rtl_sources=rtl_cfg.get("sources", []),
            include_dirs=rtl_cfg.get("include_dirs", []),
            generated_sv=[],
            vten_sv_dir="",
            dpi_c_source="",
            compile_options=xsim_cfg.get("compile_options", []),
        )

        return GeneratorContext(tb=tb, build=build)

    def _compute_scheduler_params(self, num_commands: int) -> dict:
        """Compute Scheduler parameters from BFM topology and command count."""
        auto_bfms = max(8, len(self.bfm_configs))
        auto_ifaces = max(16, len(self.spec.interface_names()))
        auto_cmds = max(256, num_commands)

        sched_cfg = self.config.get("backend", {}).get("scheduler", {})
        return {
            "max_bfms": max(auto_bfms, sched_cfg.get("max_bfms", 0)),
            "max_ifaces": max(auto_ifaces, sched_cfg.get("max_ifaces", 0)),
            "max_cmds": max(auto_cmds, sched_cfg.get("max_cmds", 0)),
        }

    def _generate_bfm_index_mapping(self) -> dict[int, int]:
        """Generate interface_id → BFM index lookup table."""
        iface_names = self.spec.interface_names()
        iface_id_map = {name: idx for idx, name in enumerate(iface_names)}
        bfm_idx_map = {cfg.interface_name: idx for idx, cfg in enumerate(self.bfm_configs)}

        mapping: dict[int, int] = {}
        for name, iface_id in iface_id_map.items():
            if name in bfm_idx_map:
                mapping[iface_id] = bfm_idx_map[name]
        return mapping

    def generate(self, output_dir: str, num_commands: int = 0) -> None:
        """Generate testbench, build scripts, run scripts to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ctx = self._build_context()
        sched_params = self._compute_scheduler_params(num_commands=num_commands)
        iface_to_bfm = self._generate_bfm_index_mapping()

        template_dir = Path(__file__).resolve().parent.parent.parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
        )

        # Resolve vten_sv_dir: either from build context or auto-detect
        vten_sv_dir = ctx.build.vten_sv_dir
        if not vten_sv_dir:
            candidate = Path(__file__).resolve().parent.parent.parent / "vten_sv"
            if candidate.exists():
                vten_sv_dir = str(candidate)
            else:
                vten_sv_dir = "vten_sv"

        template_vars = {
            "tb": ctx.tb,
            "build": ctx.build,
            "bfms": ctx.tb.bfms,
            "top_module": ctx.tb.top_module,
            "clock_name": ctx.tb.clock_name,
            "reset_name": ctx.tb.reset_name,
            "reset_active_low": ctx.tb.reset_active_low,
            "clock_period_ns": ctx.tb.clock_period_ns,
            "timeout_cycles": ctx.tb.timeout_cycles,
            "max_cmds": sched_params["max_cmds"],
            "max_bfms": sched_params["max_bfms"],
            "max_ifaces": sched_params["max_ifaces"],
            "iface_to_bfm": iface_to_bfm,
            "rtl_sources": ctx.build.rtl_sources,
            "compile_options": ctx.build.compile_options,
            "vivado_path": ctx.build.vivado_path,
            "timescale": ctx.build.timescale,
            "vten_sv_dir": vten_sv_dir,
            "generated_dir": str(out),
            "lib_dir": str(out.parent / "lib") if out.parent.exists() else "build/lib",
        }

        file_map = {
            "tb_top.sv.j2": "tb_top.sv",
            "build_xsim.tcl.j2": "build.tcl",
            "run_xsim.tcl.j2": "run.tcl",
            "Makefile.j2": "Makefile",
        }

        for template_name, output_name in file_map.items():
            tmpl = env.get_template(template_name)
            rendered = tmpl.render(**template_vars)
            (out / output_name).write_text(rendered)
