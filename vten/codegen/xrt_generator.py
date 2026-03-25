"""XRT code generator — generates TCL, XML, and config files for xclbin build.

Produces:
  - package_ip.tcl: Vivado IP Packager TCL
  - gen_xo.tcl: XO creation TCL
  - kernel.xml: Kernel metadata XML
  - connectivity.cfg: v++ link configuration

Spec reference: 08_backend_abstraction.md §11
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from vten.spec.models import KernelSpec, Protocol


def _vten_templates_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "templates"


def _render(template_name: str, context: dict) -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_vten_templates_dir())),
    )
    return env.get_template(template_name).render(context)


def _interfaces_context(spec: KernelSpec) -> list[dict]:
    """Build template-friendly interface list from KernelSpec."""
    result = []
    for name, iface in spec.interfaces.items():
        entry = {
            "name": name,
            "protocol": iface.protocol.value,
            "data_width": iface.data_width or 256,
            "addr_width": iface.addr_width or 64,
            "rtl_port": iface.rtl_port,
        }
        # Direction inference for AXI4-Stream
        if iface.protocol == Protocol.AXI4S:
            if iface.rtl_port.startswith("s_"):
                entry["direction"] = "input"
            else:
                entry["direction"] = "output"

        # XRT memory bank
        if iface.xrt and iface.xrt.memory_bank:
            entry["memory_bank"] = iface.xrt.memory_bank
        elif iface.memory_region:
            entry["memory_bank"] = iface.memory_region

        # Buffer size estimation
        entry["buffer_size"] = 4096  # default, overridden by actual tensor size

        # AXI4-Lite address range
        if iface.protocol == Protocol.AXI4L:
            entry["addr_range"] = 2 ** (iface.addr_width or 12)

        result.append(entry)
    return result


def _registers_context(spec: KernelSpec) -> list[dict]:
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

        # Build template contexts
        interfaces = {}
        for name, iface in self._spec.interfaces.items():
            interfaces[name] = {
                "protocol": iface.protocol.value,
                "data_width": iface.data_width or 256,
                "addr_width": iface.addr_width or 64,
                "rtl_port": iface.rtl_port,
                "direction": "input" if (
                    iface.protocol == Protocol.AXI4S
                    and iface.rtl_port.startswith("s_")
                ) else "output",
                "memory_bank": (
                    iface.xrt.memory_bank
                    if iface.xrt and iface.xrt.memory_bank
                    else iface.memory_region or "HBM[0]"
                ),
                "buffer_size": 4096,
                "addr_range": 2 ** (iface.addr_width or 12) if iface.protocol == Protocol.AXI4L else 0,
            }

        rtl_sources = []
        for pat in self._config.get("rtl", {}).get("sources", []):
            rtl_sources.append(pat)

        registers = _registers_context(self._spec)

        kernel_name = self._spec.kernel_name
        generated: dict[str, Path] = {}

        # 1. package_ip.tcl
        packaging_dir = out / "packaging"
        packaging_dir.mkdir(exist_ok=True)

        package_ip_tcl = _render("package_ip.tcl.j2", {
            "kernel_name": kernel_name,
            "rtl_sources": rtl_sources,
            "interfaces": interfaces,
            "registers": registers,
        })
        path = packaging_dir / "package_ip.tcl"
        path.write_text(package_ip_tcl)
        generated["package_ip.tcl"] = path

        # 2. kernel.xml
        kernel_xml = _render("kernel.xml.j2", {
            "kernel_name": kernel_name,
            "interfaces": interfaces,
        })
        path = packaging_dir / "kernel.xml"
        path.write_text(kernel_xml)
        generated["kernel.xml"] = path

        # 3. gen_xo.tcl
        xo_path = f"{kernel_name}.xo"
        gen_xo_tcl = _render("gen_xo.tcl.j2", {
            "kernel_name": kernel_name,
            "xo_path": xo_path,
            "kernel_xml_path": "packaging/kernel.xml",
        })
        path = packaging_dir / "xo_gen.tcl"
        path.write_text(gen_xo_tcl)
        generated["xo_gen.tcl"] = path

        # 4. connectivity.cfg
        link_dir = out / "link"
        link_dir.mkdir(exist_ok=True)

        connectivity_cfg = _render("connectivity.cfg.j2", {
            "kernel_name": kernel_name,
            "interfaces": interfaces,
            "stream_connections": [],
        })
        path = link_dir / "connectivity.cfg"
        path.write_text(connectivity_cfg)
        generated["connectivity.cfg"] = path

        return generated
