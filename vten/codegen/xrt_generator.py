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


def _flat_index_tuple(dimensions: list[int], flat_idx: int) -> tuple[int, ...]:
    """Convert flat index to multi-dimensional index tuple.

    E.g. dimensions=[32, 2], flat_idx=5 → (2, 1)
    """
    indices: list[int] = []
    for d in reversed(dimensions):
        indices.append(flat_idx % d)
        flat_idx //= d
    return tuple(reversed(indices))


def _expand_bank_pattern(pattern: str, indices: tuple[int, ...]) -> str:
    """Expand memory bank pattern with index variables.

    E.g. "HBM[{i}]", (5,) → "HBM[5]"
         "DDR[0]", (0,) → "DDR[0]" (no pattern variable, unchanged)
    """
    var_names = "ijklmn"
    fmt = {var_names[i]: v for i, v in enumerate(indices)}
    try:
        return pattern.format(**fmt)
    except (KeyError, IndexError):
        return pattern


def _flat_ext_port(iface, flat_name: str) -> str:
    """Compute Vitis-compatible ext_port for a flat array element.

    E.g. AXI4 master "wgt_dma_0" → "m_axi_wgt_dma_0"
         AXIS slave "weight_0_0" → "s_axis_weight_0_0"
    """
    _role = iface.role or (
        "master" if iface.rtl_port.startswith("m_") else "slave"
    )
    if iface.protocol == Protocol.AXI4L:
        return f"s_axi_{flat_name}"
    elif iface.protocol == Protocol.AXI4S:
        prefix = "m_axis" if _role == "master" else "s_axis"
        return f"{prefix}_{flat_name}"
    elif iface.protocol == Protocol.AXI4:
        prefix = "m_axi" if _role == "master" else "s_axi"
        return f"{prefix}_{flat_name}"
    return flat_name


def _build_interfaces_context(spec: KernelSpec) -> dict[str, dict]:
    """Build template-friendly interfaces dict from KernelSpec.

    Array interfaces are flattened to individual elements, each with
    its own ext_port and memory_bank mapping.
    """
    interfaces = {}
    for name, iface in spec.interfaces.items():
        # Shared base properties
        base = {
            "protocol": iface.protocol.value,
            "data_width": iface.data_width or 256,
            "addr_width": iface.addr_width or 64,
            "rtl_port": iface.rtl_port,
            "buffer_size": 4096,
        }

        # Direction inference
        if iface.protocol == Protocol.AXI4S:
            role = iface.role or (
                "master" if iface.rtl_port.startswith("m_") else "slave"
            )
            base["direction"] = "output" if role == "master" else "input"
        else:
            base["direction"] = "output"

        # AXI4-Lite address range
        if iface.protocol == Protocol.AXI4L:
            base["addr_range"] = 2 ** (iface.addr_width or 12)

        # Memory bank pattern
        if iface.xrt and iface.xrt.memory_bank:
            bank_pattern = iface.xrt.memory_bank
        elif iface.memory_region:
            bank_pattern = iface.memory_region
        else:
            bank_pattern = "DDR[0]"

        if iface.array:
            # Flatten array to individual elements
            flat_names = iface.array.flat_names(name)
            for idx, flat_name in enumerate(flat_names):
                entry = dict(base)
                entry["name"] = flat_name
                entry["ext_port"] = _flat_ext_port(iface, flat_name)
                indices = _flat_index_tuple(iface.array.dimensions, idx)
                entry["memory_bank"] = _expand_bank_pattern(bank_pattern, indices)
                entry["parent_name"] = name
                entry["array_index"] = idx
                interfaces[flat_name] = entry
        else:
            entry = dict(base)
            entry["name"] = name
            entry["ext_port"] = iface.ext_port
            entry["memory_bank"] = bank_pattern
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
    """Build kernel.xml args from interfaces.

    AXI4 memory interfaces with xrt.arg_index become memory args (addrQual=1).
    AXI-Stream interfaces become stream args (addrQual=4) — required for
    v++ stream_connect to discover ports.
    Array interfaces are expanded: each element gets its own arg with
    sequential arg_index.
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

    # Auto-assign arg IDs: explicit xrt.arg_index first, then streams
    next_id = 0

    # Pass 1: AXI4 interfaces with explicit xrt.arg_index
    for name, iface in spec.interfaces.items():
        if not iface.xrt or iface.xrt.arg_index is None:
            continue

        tensor_name = getattr(iface, "tensor", None) or name
        offset = tensor_addr_offsets.get(tensor_name, 0x10)
        base_id = iface.xrt.arg_index

        if iface.array:
            flat_names = iface.array.flat_names(name)
            for idx, flat_name in enumerate(flat_names):
                arg_id = base_id + idx
                args.append({
                    "name": flat_name,
                    "address_qualifier": 1,
                    "id": arg_id,
                    "port": _flat_ext_port(iface, flat_name),
                    "size": 8,
                    "offset": offset,
                    "type": "int*",
                })
                next_id = max(next_id, arg_id + 1)
        else:
            args.append({
                "name": name,
                "address_qualifier": 1,
                "id": base_id,
                "port": iface.ext_port,
                "size": 8,
                "offset": offset,
                "type": "int*",
            })
            next_id = max(next_id, base_id + 1)

    # Pass 1b: AXI-Lite scalar arg (ensures v++ assigns base address)
    # Kernels without AXI-MM args but with AXI-Lite ctrl need at least one
    # scalar arg so v++ allocates a base address for register access.
    has_mm_args = any(a["address_qualifier"] == 1 for a in args)
    for name, iface in spec.interfaces.items():
        if iface.protocol != Protocol.AXI4L:
            continue
        if not has_mm_args:
            args.append({
                "name": "ctrl",
                "address_qualifier": 0,
                "id": next_id,
                "port": iface.ext_port,
                "size": 4,
                "offset": 0x10,
                "type": "uint32_t",
            })
            next_id += 1
        break  # only one AXI-Lite ctrl per kernel

    # Pass 2: AXI-Stream interfaces (stream args for v++ stream_connect)
    for name, iface in spec.interfaces.items():
        if iface.protocol != Protocol.AXI4S:
            continue

        if iface.array:
            flat_names = iface.array.flat_names(name)
            for idx, flat_name in enumerate(flat_names):
                args.append({
                    "name": flat_name,
                    "address_qualifier": 4,
                    "id": next_id,
                    "port": _flat_ext_port(iface, flat_name),
                    "size": 8,
                    "offset": 0,
                    "type": "int*",
                })
                next_id += 1
        else:
            args.append({
                "name": name,
                "address_qualifier": 4,
                "id": next_id,
                "port": iface.ext_port,
                "size": 8,
                "offset": 0,
                "type": "int*",
            })
            next_id += 1

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

        # Control protocol: user_managed (default for RTL kernels, no ap_ctrl_hs)
        ctrl_protocol = xrt_config.get("ctrl_protocol", "user_managed")

        generated: dict[str, Path] = {}

        # IP entries from build pipeline
        ip_sources = self._config.get("_ip_sources", [])
        ip_create = self._config.get("_ip_create", [])

        # Include directories for headers (.svh, .vh)
        include_dirs = self._config.get("rtl", {}).get("include_dirs", [])

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
            "include_dirs": include_dirs,
            "ctrl_protocol": ctrl_protocol,
        })
        path = out / "package_ip.tcl"
        path.write_text(content)
        generated["package_ip.tcl"] = path

        # 2. kernel.xml
        content = _render("kernel.xml.j2", {
            "kernel_name": kernel_name,
            "interfaces": interfaces,
            "args": args,
            "ctrl_protocol": ctrl_protocol,
        })
        path = out / "kernel.xml"
        path.write_text(content)
        generated["kernel.xml"] = path

        # 3. gen_xo.tcl
        content = _render("gen_xo.tcl.j2", {
            "kernel_name": kernel_name,
            "ctrl_protocol": ctrl_protocol,
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
