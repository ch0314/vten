"""SVGenerator: Jinja2-based SystemVerilog testbench code generation.

Spec reference: 06_codegen_and_cli.md §1-3
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import jinja2

from itertools import product

from vten.runtime.ir import BFMConfig
from vten.spec.models import ArraySpec, InterfaceSpec, KernelSpec, Protocol


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
        probe_bfms: list[dict] | None = None,
    ) -> None:
        self.spec = kernel_spec
        self.bfm_configs = bfm_configs
        self.config = project_config
        self.probe_bfms = probe_bfms or []

    def _module_for_protocol(self, protocol: Protocol) -> str:
        return _PROTOCOL_MODULE_MAP[protocol]

    @staticmethod
    def _flat_ext_port_for_element(
        iface: InterfaceSpec, logical_name: str, flat_name: str
    ) -> str:
        """Derive flat Vitis-compatible port prefix for an array element.

        E.g. iface.name="wgt", flat_name="wgt_0_1", protocol=AXI4S slave
        → "s_axis_wgt_0_1"
        """
        prefix_map = {
            (Protocol.AXI4S, "master"): "m_axis_",
            (Protocol.AXI4S, "slave"): "s_axis_",
            (Protocol.AXI4, "master"): "m_axi_",
            (Protocol.AXI4, "slave"): "s_axi_",
        }
        role = iface.role or ("master" if iface.rtl_port.startswith("m_") else "slave")
        prefix = prefix_map.get((iface.protocol, role), iface.rtl_port + "_")
        return prefix + flat_name

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
                addr_w = bfm.parameters.get("ADDR_W", 32) if hasattr(bfm, "parameters") else 32
                signals = [
                    (f"{prefix}_awaddr", "input", addr_w),
                    (f"{prefix}_awvalid", "input", 1),
                    (f"{prefix}_awready", "output", 1),
                    (f"{prefix}_wdata", "input", bfm.data_width),
                    (f"{prefix}_wstrb", "input", bfm.data_width // 8),
                    (f"{prefix}_wvalid", "input", 1),
                    (f"{prefix}_wready", "output", 1),
                    (f"{prefix}_bresp", "output", 2),
                    (f"{prefix}_bvalid", "output", 1),
                    (f"{prefix}_bready", "input", 1),
                    (f"{prefix}_araddr", "input", addr_w),
                    (f"{prefix}_arvalid", "input", 1),
                    (f"{prefix}_arready", "output", 1),
                    (f"{prefix}_rdata", "output", bfm.data_width),
                    (f"{prefix}_rresp", "output", 2),
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
            iface, logical_name = self.spec.resolve_flat_interface(
                cfg.interface_name
            )
            params: dict = {"DATA_W": cfg.data_width}
            if cfg.protocol == Protocol.AXI4:
                params["ADDR_W"] = cfg.addr_width
            elif cfg.protocol == Protocol.AXI4S:
                params["MODE"] = '"MASTER"' if cfg.role == "master" else '"SLAVE"'
            elif cfg.protocol == Protocol.AXI4L:
                params["ADDR_W"] = cfg.addr_width or 32

            # Port prefix: wrapper uses ext_port (Vitis naming), raw RTL uses rtl_port
            # Composite kernels have no generate_controller but DO have a wrapper
            # (rtl_top="" indicates auto-generated composite wrapper)
            has_wrapper = self._has_generate_controller() or not self.spec.rtl_top
            if iface.array and cfg.interface_name != logical_name:
                port_prefix = self._flat_ext_port_for_element(
                    iface, logical_name, cfg.interface_name
                )
            elif has_wrapper:
                port_prefix = iface.ext_port
            else:
                port_prefix = iface.rtl_port

            bfms.append(BFMInstance(
                name=f"bfm_{cfg.interface_name}",
                module_name=self._module_for_protocol(cfg.protocol),
                protocol=cfg.protocol.value,
                data_width=cfg.data_width,
                role=cfg.role,
                rtl_port_prefix=port_prefix,
                parameters=params,
                interface_id=i,
            ))

        rtl_cfg = self.config.get("rtl", {})
        # DUT module name resolution priority:
        # 0. If wrapper is generated, DUT is the wrapper (kernel_name)
        # 1. rtl.top_module from config (explicit)
        # 2. Derived from spec.rtl_top filename stem
        # 3. spec.kernel_name as fallback
        if has_wrapper:
            top_module = self.spec.kernel_name
        else:
            top_module = rtl_cfg.get("top_module", "")
            if not top_module and self.spec.rtl_top:
                top_module = Path(self.spec.rtl_top).stem
            if not top_module:
                top_module = self.spec.kernel_name or ""

        # Derive DUT ports from BFM signal topology
        dut_ports = self._derive_dut_ports(bfms)

        # When wrapper is generated, tb_top connects to wrapper using ap_clk/ap_aresetn.
        # When no wrapper, tb_top connects directly to DUT using its native clock/reset names.
        if self._has_generate_controller():
            tb_clock = "ap_clk"
            tb_reset = "ap_aresetn"
        else:
            tb_clock = self.spec.clock_name
            tb_reset = self.spec.reset_name

        tb = TestbenchContext(
            project_name=self.config.get("project", {}).get("name", ""),
            top_module=top_module,
            session_id=uuid.uuid4().hex[:16],
            dut_ports=dut_ports,
            bfms=bfms,
            clock_name=tb_clock,
            reset_name=tb_reset,
            reset_active_low=self.spec.reset_active_low,
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
        auto_ifaces = max(16, len(self.spec.expanded_interface_names()))
        auto_cmds = max(256, num_commands)

        sched_cfg = self.config.get("backend", {}).get("scheduler", {})
        return {
            "max_bfms": max(auto_bfms, sched_cfg.get("max_bfms", 0)),
            "max_ifaces": max(auto_ifaces, sched_cfg.get("max_ifaces", 0)),
            "max_cmds": max(auto_cmds, sched_cfg.get("max_cmds", 0)),
        }

    def _generate_bfm_index_mapping(self) -> dict[int, int]:
        """Generate interface_id → BFM index lookup table.

        Uses expanded_interface_names() to match IR lowering's ID assignment.
        """
        expanded = self.spec.expanded_interface_names()
        iface_id_map = {name: idx for idx, name in enumerate(expanded)}
        bfm_idx_map = {cfg.interface_name: idx for idx, cfg in enumerate(self.bfm_configs)}

        mapping: dict[int, int] = {}
        for name, iface_id in iface_id_map.items():
            if name in bfm_idx_map:
                mapping[iface_id] = bfm_idx_map[name]
        return mapping

    def _has_generate_controller(self) -> bool:
        """Check if any interface requires controller generation."""
        return any(
            iface.generate_controller
            for iface in self.spec.interfaces.values()
        )

    def _classify_interfaces(self) -> tuple[
        list[InterfaceSpec], list[InterfaceSpec], list[InterfaceSpec]
    ]:
        """Classify interfaces into ctrl, stream, aximm for wrapper generation.

        Array interfaces are excluded from stream/aximm lists (handled separately).
        """
        ctrl, stream, aximm = [], [], []
        for iface in self.spec.interfaces.values():
            if iface.protocol == Protocol.AXI4L and iface.generate_controller:
                ctrl.append(iface)
            elif iface.array:
                continue  # handled by _expand_array_interfaces
            elif iface.protocol == Protocol.AXI4S:
                stream.append(iface)
            elif iface.protocol == Protocol.AXI4:
                aximm.append(iface)
        return ctrl, stream, aximm

    @staticmethod
    def _expand_array_interfaces(
        interfaces: dict[str, InterfaceSpec],
    ) -> list[dict]:
        """Expand array interfaces into flat element descriptors for the template.

        Returns a list of dicts, one per array interface, each containing:
          - iface: the InterfaceSpec
          - sv_array_dims: "[32][2]" string for SV declaration
          - elements: list of {flat_name, indices_str} for each element
        """
        result = []
        for iface in interfaces.values():
            if not iface.array:
                continue
            arr: ArraySpec = iface.array
            dims = arr.dimensions

            # SV array dimension string: [32][2]
            sv_dims = "".join(f"[{d}]" for d in dims)

            # Resolve flat_name_pattern: explicit or auto from name
            pattern = arr.flat_name_pattern
            if not pattern:
                var_names = "ijklmn"
                pattern = iface.name + "".join(
                    f"_{{{var_names[d]}}}" for d in range(len(dims))
                )

            # Expand all index combinations
            ranges = [range(d) for d in dims]
            elements = []
            for indices in product(*ranges):
                # flat name: pattern.format(i=0, j=1, ...)
                idx_vars = {}
                var_names = "ijklmn"
                for vi, val in enumerate(indices):
                    idx_vars[var_names[vi]] = val
                flat_name = pattern.format(**idx_vars)

                # SV index string: [0][1]
                indices_str = "".join(f"[{i}]" for i in indices)

                elements.append({
                    "flat_name": flat_name,
                    "indices_str": indices_str,
                })

            # Determine role/direction for port generation
            role = iface.role
            if not role:
                # Infer from rtl_port prefix
                if iface.rtl_port.startswith("m_"):
                    role = "master"
                else:
                    role = "slave"

            # rtl_port prefix for flat ports
            prefix_map = {
                (Protocol.AXI4S, "master"): "m_axis_",
                (Protocol.AXI4S, "slave"): "s_axis_",
                (Protocol.AXI4, "master"): "m_axi_",
                (Protocol.AXI4, "slave"): "s_axi_",
            }
            flat_prefix = prefix_map.get(
                (iface.protocol, role), iface.rtl_port + "_"
            )

            result.append({
                "iface": iface,
                "sv_array_dims": sv_dims,
                "elements": elements,
                "role": role,
                "flat_prefix": flat_prefix,
            })
        return result

    def _generate_axilite_ctrl(
        self, env: jinja2.Environment, out: Path, iface: InterfaceSpec
    ) -> str:
        """Generate AXI-Lite controller module. Returns output filename."""
        tmpl = env.get_template("axilite_ctrl.sv.j2")
        rendered = tmpl.render(
            kernel_name=self.spec.kernel_name,
            iface_name=iface.name,
            addr_width=iface.addr_width or 32,
            data_width=iface.data_width or 32,
            registers=iface.registers or [],
            clock_name="ap_clk",
            reset_name="ap_aresetn",
            reset_active_low=self.spec.reset_active_low,
        )
        filename = f"{self.spec.kernel_name}_axilite_ctrl.sv"
        (out / filename).write_text(rendered)
        return filename

    @staticmethod
    def _build_wrapper_parameters(
        spec_params: dict[str, str | int],
        interfaces: dict[str, "InterfaceSpec"],
    ) -> dict[str, int]:
        """Build wrapper module parameters from spec params + interface widths.

        Auto-generates DATA_W / ADDR_W parameters for AXI4 and AXI4-Stream
        interfaces so Vivado can infer address space ranges from parameters.

        Naming: if all non-ctrl interfaces share the same width, use
        ``DATA_W`` / ``ADDR_W``.  Otherwise, use per-interface names
        ``<NAME>_DATA_W`` / ``<NAME>_ADDR_W`` (NAME = upper-cased iface name).
        """
        from vten.spec.models import Protocol

        # Filter out unresolved template references (${...}) — they are
        # runtime parameters set via registers, not compile-time constants.
        params: dict[str, int] = {
            k: v for k, v in spec_params.items()
            if not (isinstance(v, str) and "${" in v)
        }  # type: ignore[arg-type]

        # Collect unique data/addr widths across non-ctrl interfaces
        bus_ifaces = {
            name: iface
            for name, iface in interfaces.items()
            if iface.protocol in (Protocol.AXI4, Protocol.AXI4S)
        }

        if not bus_ifaces:
            return params

        data_widths = {
            name: iface.data_width
            or (iface.packing.bus_width if getattr(iface, "packing", None) else 256)
            for name, iface in bus_ifaces.items()
        }
        addr_widths = {
            name: iface.addr_width or 64
            for name, iface in bus_ifaces.items()
            if iface.protocol == Protocol.AXI4
        }

        unique_dw = set(data_widths.values())
        unique_aw = set(addr_widths.values())

        # Assign parameter names
        if len(unique_dw) == 1 and "DATA_W" not in params:
            shared_dw_name = "DATA_W"
            for name in bus_ifaces:
                bus_ifaces[name]._wrapper_data_w_param = shared_dw_name  # type: ignore[attr-defined]
            params[shared_dw_name] = next(iter(unique_dw))
        else:
            for name, dw in data_widths.items():
                pname = f"{name.upper()}_DATA_W"
                bus_ifaces[name]._wrapper_data_w_param = pname  # type: ignore[attr-defined]
                params.setdefault(pname, dw)

        if len(unique_aw) == 1 and "ADDR_W" not in params:
            shared_aw_name = "ADDR_W"
            for name in addr_widths:
                bus_ifaces[name]._wrapper_addr_w_param = shared_aw_name  # type: ignore[attr-defined]
            params[shared_aw_name] = next(iter(unique_aw))
        else:
            for name, aw in addr_widths.items():
                pname = f"{name.upper()}_ADDR_W"
                bus_ifaces[name]._wrapper_addr_w_param = pname  # type: ignore[attr-defined]
                params.setdefault(pname, aw)

        return params

    def _generate_wrapper(self, env: jinja2.Environment, out: Path) -> str:
        """Generate wrapper module. Returns output filename."""
        ctrl, stream, aximm = self._classify_interfaces()
        all_ifaces = list(self.spec.interfaces.values())
        array_groups = self._expand_array_interfaces(self.spec.interfaces)

        # Build parameters with auto-generated DATA_W / ADDR_W
        wrapper_params = self._build_wrapper_parameters(
            self.spec.parameters, self.spec.interfaces,
        )

        tmpl = env.get_template("wrapper.sv.j2")
        rendered = tmpl.render(
            kernel_name=self.spec.kernel_name,
            parameters=wrapper_params,
            clock_name="ap_clk",
            reset_name="ap_aresetn",
            core_clock_name=self.spec.clock_name,
            core_reset_name=self.spec.reset_name,
            reset_active_low=self.spec.reset_active_low,
            all_interfaces=all_ifaces,
            ctrl_interfaces=ctrl,
            stream_interfaces=stream,
            aximm_interfaces=aximm,
            array_groups=array_groups,
        )
        filename = f"{self.spec.kernel_name}_wrapper.sv"
        (out / filename).write_text(rendered)

        # Clean up temporary attrs to avoid polluting shared InterfaceSpec objects
        for iface in self.spec.interfaces.values():
            for attr in ("_wrapper_data_w_param", "_wrapper_addr_w_param"):
                if hasattr(iface, attr):
                    delattr(iface, attr)

        return filename

    def generate(self, output_dir: str, num_commands: int = 0) -> list[str]:
        """Generate testbench and optional controller/wrapper to output_dir.

        Returns list of generated filenames.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ctx = self._build_context()
        sched_params = self._compute_scheduler_params(num_commands=num_commands)
        iface_to_bfm = self._generate_bfm_index_mapping()

        template_dir = Path(__file__).resolve().parent.parent / "templates" / "sim"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
        )

        # Resolve vten_sv_dir: either from build context or auto-detect
        vten_sv_dir = ctx.build.vten_sv_dir
        if not vten_sv_dir:
            candidate = Path(__file__).resolve().parent.parent / "sv"
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
            "probe_bfms": self.probe_bfms,
        }

        generated_files = []

        # Testbench
        tmpl = env.get_template("tb_top.sv.j2")
        rendered = tmpl.render(**template_vars)
        (out / "tb_top.sv").write_text(rendered)
        generated_files.append("tb_top.sv")

        # Waveform TCL (always generated at build time; used at runtime if --waveform)
        tmpl_wave = env.get_template("waveform.tcl.j2")
        rendered_wave = tmpl_wave.render(**template_vars)
        (out / "waveform.tcl").write_text(rendered_wave)
        generated_files.append("waveform.tcl")

        # Controller + Wrapper (if any interface has generate_controller)
        if self._has_generate_controller():
            for iface in self.spec.interfaces.values():
                if iface.generate_controller:
                    fname = self._generate_axilite_ctrl(env, out, iface)
                    generated_files.append(fname)
            fname = self._generate_wrapper(env, out)
            generated_files.append(fname)

        return generated_files
