"""XRT code generator — generates TCL, XML, and config files for xclbin build.

Produces:
  - package_ip.tcl: Vivado IP Packager TCL
  - gen_xo.tcl: XO creation TCL
  - kernel.xml: Kernel metadata XML
  - connectivity.cfg: v++ link configuration
  - build_{target}.sh: Build script

Spec reference: 08_backend_abstraction.md §11
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from vten.spec.models import KernelSpec, Protocol


def _vten_templates_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "templates"


def _vten_sv_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "vten_sv"


def _render(template_name: str, context: dict) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_vten_templates_dir())),
    )
    return env.get_template(template_name).render(context)


def _build_interfaces_context(spec: KernelSpec) -> dict[str, dict]:
    """Build template-friendly interfaces dict from KernelSpec."""
    interfaces = {}
    for name, iface in spec.interfaces.items():
        entry = {
            "name": name,
            "protocol": iface.protocol.value,
            "data_width": iface.data_width or 256,
            "addr_width": iface.addr_width or 64,
            "rtl_port": iface.rtl_port,
            "ext_port": iface.ext_port,
        }
        # Direction inference for AXI4-Stream
        if iface.protocol == Protocol.AXI4S:
            role = iface.role or (
                "master" if iface.rtl_port.startswith("m_") else "slave"
            )
            entry["direction"] = "output" if role == "master" else "input"
        else:
            entry["direction"] = "output"

        # Memory bank
        if iface.xrt and iface.xrt.memory_bank:
            entry["memory_bank"] = iface.xrt.memory_bank
        elif iface.memory_region:
            entry["memory_bank"] = iface.memory_region
        else:
            entry["memory_bank"] = "DDR[0]"

        # Buffer size
        entry["buffer_size"] = 4096

        # AXI4-Lite address range
        if iface.protocol == Protocol.AXI4L:
            entry["addr_range"] = 2 ** (iface.addr_width or 12)

        interfaces[name] = entry
    return interfaces


def _build_registers_context(spec: KernelSpec) -> list[dict]:
    """Extract register definitions for IP packaging."""
    registers = []
    for name, iface in spec.interfaces.items():
        if iface.protocol != Protocol.AXI4L:
            continue
        if iface.registers:
            for reg in iface.registers:
                registers.append({
                    "name": reg.name,
                    "offset": reg.offset,
                    "width": reg.width,
                })
    return registers


def _build_args_context(spec: KernelSpec) -> list[dict]:
    """Build kernel.xml args from interfaces with xrt.arg_index.

    Each AXI4 memory interface with xrt config becomes an arg.
    The offset is derived from the first auto_bind register with value=address
    for the corresponding tensor.
    """
    args = []

    # Map tensor name → first address register offset
    tensor_addr_offsets: dict[str, int] = {}
    for _name, iface in spec.interfaces.items():
        if iface.protocol != Protocol.AXI4L or not iface.registers:
            continue
        for reg in iface.registers:
            if (
                reg.auto_bind
                and reg.auto_bind.value == "address"
                and reg.auto_bind.tensor
                and reg.auto_bind.tensor not in tensor_addr_offsets
            ):
                tensor_addr_offsets[reg.auto_bind.tensor] = reg.offset

    # Build args from AXI4 interfaces with xrt config
    for name, iface in spec.interfaces.items():
        if not iface.xrt or iface.xrt.arg_index is None:
            continue

        tensor_name = getattr(iface, "tensor", None) or name
        offset = tensor_addr_offsets.get(tensor_name, 0x10)

        args.append({
            "name": name,
            "address_qualifier": 4 if iface.protocol == Protocol.AXI4S else 1,
            "id": iface.xrt.arg_index,
            "port": iface.ext_port,
            "size": 8,  # 64-bit address pointer
            "offset": offset,
            "type": "int*",
        })

    args.sort(key=lambda a: a["id"])
    return args


class XrtGenerator:
    """Generates XRT build artifacts from KernelSpec.

    Usage:
        gen = XrtGenerator(kernel_spec, project_config)
        gen.generate(output_dir)
    """

    def __init__(
        self,
        kernel_spec: KernelSpec,
        project_config: dict | None = None,
    ) -> None:
        self._spec = kernel_spec
        self._config = project_config or {}

    def generate(self, output_dir: str) -> dict[str, Path]:
        """Generate all XRT build artifacts.

        Args:
            output_dir: Directory to write generated files.

        Returns:
            Dict mapping artifact name to file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        interfaces = _build_interfaces_context(self._spec)
        registers = _build_registers_context(self._spec)
        args = _build_args_context(self._spec)
        kernel_name = self._spec.kernel_name

        # RTL sources from project config (relative to project_root)
        rtl_sources = self._config.get("rtl", {}).get("sources", [])

        # Generated codegen files — filenames only (relative to $project_dir)
        generated_files = self._config.get("generated_files", [])

        # vten_sv interface files — filenames only (relative to $vten_root)
        vten_sv_files = self._config.get("vten_sv_files", [])

        # Project root (relative path from output_dir to project root)
        project_root = self._config.get("_project_root", "")

        # vten_root (path to vten_sv directory)
        vten_root = self._config.get("_vten_root", "")

        # FPGA part
        part = self._config.get("backend", {}).get("xrt", {}).get("part", "")

        # Platform and target for build script
        xrt_config = self._config.get("backend", {}).get("xrt", {})
        platform = xrt_config.get("platform", "")
        target = xrt_config.get("target", "hw_emu")

        generated: dict[str, Path] = {}

        # IP entries from build pipeline
        ip_sources = self._config.get("_ip_sources", [])
        ip_create = self._config.get("_ip_create", [])

        # 1. package_ip.tcl
        content = _render("package_ip.tcl.j2", {
            "kernel_name": kernel_name,
            "rtl_sources": rtl_sources,
            "generated_files": generated_files,
            "vten_sv_files": vten_sv_files,
            "project_root": project_root,
            "vten_root": vten_root,
            "interfaces": interfaces,
            "registers": registers,
            "part": part,
            "ip_sources": ip_sources,
            "ip_create": ip_create,
        })
        path = out / "package_ip.tcl"
        path.write_text(content)
        generated["package_ip.tcl"] = path

        # 2. kernel.xml
        content = _render("kernel.xml.j2", {
            "kernel_name": kernel_name,
            "interfaces": interfaces,
            "args": args,
        })
        path = out / "kernel.xml"
        path.write_text(content)
        generated["kernel.xml"] = path

        # 3. gen_xo.tcl
        content = _render("gen_xo.tcl.j2", {
            "kernel_name": kernel_name,
        })
        path = out / "gen_xo.tcl"
        path.write_text(content)
        generated["gen_xo.tcl"] = path

        # 4. connectivity.cfg
        content = _render("connectivity.cfg.j2", {
            "kernel_name": kernel_name,
            "interfaces": interfaces,
            "stream_connections": [],
        })
        path = out / "connectivity.cfg"
        path.write_text(content)
        generated["connectivity.cfg"] = path

        # 5. build script
        content = _render("build_xrt.sh.j2", {
            "kernel_name": kernel_name,
            "platform": platform,
            "target": target,
        })
        script_name = f"build_{target}.sh"
        path = out / script_name
        path.write_text(content)
        path.chmod(0o755)
        generated[script_name] = path

        return generated
