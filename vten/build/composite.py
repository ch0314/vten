"""Composite kernel build support — discovery, spec synthesis, SV generation.

Implements "Vitis flow" where sub-kernels are built independently and
CompositeKernel auto-generates a wrapper-of-wrappers that instantiates
each sub-kernel's wrapper and wires internal connections.
"""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

from vten.errors import BuildError
from vten.spec.models import InterfaceSpec, KernelSpec, Protocol, RegisterBankSpec
from vten.spec.parser import parse_kernel_spec


# ── Discovery ──


def is_composite_kernel(kernel_dir: Path) -> bool:
    """Check if a kernel directory contains a CompositeKernel subclass."""
    # Fast reject: if kernel_spec.yaml exists, it's a unit kernel
    # (composite kernels in Vitis-flow have NO kernel_spec.yaml)
    if (kernel_dir / "kernel_spec.yaml").exists():
        return False
    # Look for *_kernel.py files
    for py_file in kernel_dir.glob("*_kernel.py"):
        try:
            src = py_file.read_text()
            if "CompositeKernel" in src:
                return True
        except Exception:
            continue
    return False


def load_composite_class(kernel_dir: Path) -> type:
    """Import and return the CompositeKernel subclass from a kernel directory."""
    from vten.kernel.composite import CompositeKernel

    # Add parent dir to sys.path for sub-kernel imports
    kernels_parent = str(kernel_dir.parent)
    if kernels_parent not in sys.path:
        sys.path.insert(0, kernels_parent)

    for py_file in kernel_dir.glob("*_kernel.py"):
        mod_name = py_file.stem
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if (
                isinstance(obj, type)
                and issubclass(obj, CompositeKernel)
                and obj is not CompositeKernel
            ):
                return obj

    raise BuildError(
        f"No CompositeKernel subclass found in {kernel_dir}"
    )


# ── Spec Synthesis ──


def synthesize_spec(
    composite_cls: type,
    project: Path,
    kernel_name: str,
) -> KernelSpec:
    """Synthesize a KernelSpec from sub-kernel specs + Python connectivity.

    With independent ctrl (Pattern 1):
    - Each sub-kernel's AXI-Lite ctrl becomes a separate top-level interface
      named "{sub_name}_ctrl"
    - External stream/MM interfaces are exposed with the mapped name
    - Internal interfaces are omitted (wired in SV, no BFM)
    """
    from vten.kernel.composite import Internal

    bindings = dict(composite_cls._sub_kernel_bindings)
    interfaces: dict[str, InterfaceSpec] = {}
    memory_regions: dict = {}

    for sub_name, binding in bindings.items():
        sub_spec = _load_sub_kernel_spec(binding.kernel_class, project)
        # Merge memory regions from sub-kernel specs
        for region_name, region in sub_spec.memory_regions.items():
            if region_name not in memory_regions:
                memory_regions[region_name] = region

        for sub_iface_name, mapping in binding.interface_map.items():
            sub_iface = sub_spec.interfaces.get(sub_iface_name)
            if sub_iface is None:
                raise BuildError(
                    f"Interface '{sub_iface_name}' not found in "
                    f"sub-kernel '{binding.kernel_class.__name__}'"
                )

            if isinstance(mapping, Internal):
                # Internal wire — no top-level interface
                continue

            if isinstance(mapping, str):
                # External: direct mapping to top-level interface name
                top_name = mapping
            elif isinstance(mapping, tuple) and len(mapping) == 2:
                # Register banking: ("top_name", "bank_name")
                # For independent ctrl, each sub-kernel gets its own port
                top_name = mapping[0]
                bank_name = mapping[1]
            else:
                raise BuildError(
                    f"Unknown mapping type for {sub_name}.{sub_iface_name}: "
                    f"{mapping}"
                )

            # Clone the interface spec with new name and rtl_port
            # Never generate_controller — sub-kernel wrappers already have theirs
            new_iface = deepcopy(sub_iface)
            new_iface.name = top_name
            new_iface.generate_controller = False

            if sub_iface.protocol == Protocol.AXI4L:
                new_iface.rtl_port = f"s_axilite_{sub_name}"
                # Add register bank for independent ctrl pattern
                if isinstance(mapping, tuple) and len(mapping) == 2:
                    new_iface.register_banks = [
                        RegisterBankSpec(name=bank_name, base_offset=0)
                    ]
            elif sub_iface.protocol == Protocol.AXI4S:
                # Preserve direction convention (s_axis for slave, m_axis for master)
                # determined by the original sub-kernel spec
                pass
            elif sub_iface.protocol == Protocol.AXI4:
                # AXI4 MM: use composite-unique prefix based on top_name
                new_iface.rtl_port = f"m_axi_{top_name}"
            # For stream interfaces, update tensor name from exposed_tensor_defs
            if new_iface.tensor:
                exposed_tensor_name = _find_exposed_tensor(
                    composite_cls, sub_name, new_iface.tensor
                )
                if exposed_tensor_name:
                    new_iface.tensor = exposed_tensor_name

            if top_name not in interfaces:
                interfaces[top_name] = new_iface

    return KernelSpec(
        kernel_name=kernel_name,
        rtl_top="",  # Auto-generated, no RTL file
        interfaces=interfaces,
        memory_regions=memory_regions,
    )


def _load_sub_kernel_spec(kernel_class: type, project: Path) -> KernelSpec:
    """Load a sub-kernel's kernel_spec.yaml."""
    spec_attr = getattr(kernel_class, "spec", None)
    if spec_attr is None:
        raise BuildError(
            f"Sub-kernel {kernel_class.__name__} has no 'spec' attribute"
        )
    spec_path = project / spec_attr
    if not spec_path.exists():
        raise BuildError(f"Sub-kernel spec not found: {spec_path}")
    return parse_kernel_spec(spec_path)


def _find_exposed_tensor(
    composite_cls: type, sub_name: str, sub_tensor_name: str
) -> str | None:
    """Find the top-level exposed tensor name for a sub-kernel tensor."""
    for name, edef in composite_cls._exposed_tensor_defs.items():
        if edef.origin_sub_kernel == sub_name and edef.origin_name == sub_tensor_name:
            return name
    return None


# ── Sub-kernel dependency check ──


def get_sub_kernel_names(composite_cls: type) -> list[str]:
    """Return list of sub-kernel directory names (for build dependency)."""
    names = []
    for _attr_name, binding in composite_cls._sub_kernel_bindings.items():
        spec_attr = getattr(binding.kernel_class, "spec", None)
        if spec_attr:
            # spec = "kernels/scale/kernel_spec.yaml" → "scale"
            parts = Path(spec_attr).parts
            if len(parts) >= 2:
                names.append(parts[1])
    return names


def check_sub_kernels_built(
    composite_cls: type, project: Path
) -> list[str]:
    """Check which sub-kernels need building. Returns list of unbuilt names."""
    unbuilt = []
    for name in get_sub_kernel_names(composite_cls):
        wrapper_dir = project / "kernels" / name / "build" / "generated"
        # Check for the wrapper SV file
        wrapper_files = list(wrapper_dir.glob(f"{name}_wrapper.sv")) + \
                        list(wrapper_dir.glob(f"{name}.sv"))
        if not wrapper_files and not (wrapper_dir / "tb_top.sv").exists():
            unbuilt.append(name)
    return unbuilt


# ── Composite SV Generation ──


def generate_composite_sv(
    composite_cls: type,
    synthesized_spec: KernelSpec,
    project: Path,
    output_dir: Path,
) -> None:
    """Generate composite_top.sv — wrapper-of-wrappers.

    Instantiates each sub-kernel's wrapper module and wires:
    - External interfaces to top-level ports
    - Internal connections between sub-kernel wrappers
    """
    from vten.kernel.composite import Internal

    bindings = dict(composite_cls._sub_kernel_bindings)
    connections = list(composite_cls._connections)

    # Collect sub-kernel info
    sub_kernels = []
    for sub_name, binding in bindings.items():
        sub_spec = _load_sub_kernel_spec(binding.kernel_class, project)
        sub_kernels.append({
            "name": sub_name,
            "module_name": sub_spec.kernel_name,
            "spec": sub_spec,
            "binding": binding,
        })

    # Build internal wire list from connections
    internal_wires = []
    for idx, conn in enumerate(connections):
        wire_name = f"internal_{idx}"
        # Find the source interface spec to get DATA_W
        src_sub = conn.source_sub
        src_binding = bindings[src_sub]
        src_spec = _load_sub_kernel_spec(src_binding.kernel_class, project)
        src_iface = src_spec.interfaces.get(conn.source_interface)
        data_w = 256  # default
        if src_iface and src_iface.packing:
            data_w = src_iface.packing.bus_width
        internal_wires.append({
            "name": wire_name,
            "data_w": data_w,
            "source_sub": src_sub,
            "source_iface": conn.source_interface,
            "dest_sub": conn.dest_sub,
            "dest_iface": _find_dest_interface(conn, bindings),
        })

    # Generate SV
    lines = []
    lines.append("// Auto-generated by vTen — DO NOT EDIT")
    lines.append(f"// Composite wrapper: {synthesized_spec.kernel_name}")
    lines.append(f"// Sub-kernels: {', '.join(sk['name'] for sk in sub_kernels)}")
    lines.append("")
    lines.append("`include \"vten_types.svh\"")
    lines.append("")

    # Module declaration
    lines.append(f"module {synthesized_spec.kernel_name} (")
    lines.append("    input  logic clk,")
    lines.append("    input  logic rst_n,")

    # Collect all top-level ports
    port_lines = []
    for iface_name, iface in synthesized_spec.interfaces.items():
        port_lines.extend(_generate_port_lines(iface_name, iface))

    for i, pline in enumerate(port_lines):
        comma = "," if i < len(port_lines) - 1 else ""
        lines.append(f"    {pline}{comma}")

    lines.append(");")
    lines.append("")

    # Internal wires
    if internal_wires:
        lines.append("    // ── Internal Wires ──")
        for wire in internal_wires:
            dw = wire["data_w"]
            wn = wire["name"]
            lines.append(f"    logic [{dw-1}:0] {wn}_tdata;")
            lines.append(f"    logic           {wn}_tvalid;")
            lines.append(f"    logic           {wn}_tready;")
            lines.append(f"    logic           {wn}_tlast;")
        lines.append("")

    # Sub-kernel instantiations
    for sk in sub_kernels:
        lines.append(f"    // ── Sub-kernel: {sk['name']} ({sk['module_name']}) ──")
        inst_lines = _generate_sub_kernel_instance(
            sk, synthesized_spec, bindings, internal_wires
        )
        lines.extend(f"    {l}" for l in inst_lines)
        lines.append("")

    lines.append("endmodule")
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    sv_path = output_dir / f"{synthesized_spec.kernel_name}_composite_top.sv"
    sv_path.write_text("\n".join(lines))


def _find_dest_interface(conn, bindings):
    """Find the destination interface name from connection's dest proxy."""
    dest_sub = conn.dest_sub
    dest_tensor = conn.dest_name
    dest_binding = bindings[dest_sub]
    # Find which interface this tensor belongs to
    from vten.kernel.tensor import Tensor
    for attr_name in dir(dest_binding.kernel_class):
        attr = getattr(dest_binding.kernel_class, attr_name, None)
        if isinstance(attr, Tensor) and attr_name == dest_tensor:
            return attr.interface
    return None


def _generate_port_lines(iface_name: str, iface: InterfaceSpec) -> list[str]:
    """Generate port declaration lines for a top-level interface."""
    lines = []
    if iface.protocol == Protocol.AXI4L:
        prefix = iface.rtl_port
        aw = iface.addr_width or 16
        dw = iface.data_width or 32
        lines.append(f"// {iface_name}: AXI4-Lite")
        lines.append(f"input  logic [{aw-1}:0] {prefix}_awaddr")
        lines.append(f"input  logic          {prefix}_awvalid")
        lines.append(f"output logic          {prefix}_awready")
        lines.append(f"input  logic [{dw-1}:0] {prefix}_wdata")
        lines.append(f"input  logic [{dw//8-1}:0] {prefix}_wstrb")
        lines.append(f"input  logic          {prefix}_wvalid")
        lines.append(f"output logic          {prefix}_wready")
        lines.append(f"output logic [1:0]    {prefix}_bresp")
        lines.append(f"output logic          {prefix}_bvalid")
        lines.append(f"input  logic          {prefix}_bready")
        lines.append(f"input  logic [{aw-1}:0] {prefix}_araddr")
        lines.append(f"input  logic          {prefix}_arvalid")
        lines.append(f"output logic          {prefix}_arready")
        lines.append(f"output logic [{dw-1}:0] {prefix}_rdata")
        lines.append(f"output logic [1:0]    {prefix}_rresp")
        lines.append(f"output logic          {prefix}_rvalid")
        lines.append(f"input  logic          {prefix}_rready")
    elif iface.protocol == Protocol.AXI4S:
        dw = iface.packing.bus_width if iface.packing else 256
        rp = iface.rtl_port
        lines.append(f"// {iface_name}: AXI4-Stream")
        if rp.startswith("s_"):
            lines.append(f"input  logic [{dw-1}:0] {rp}_tdata")
            lines.append(f"input  logic          {rp}_tvalid")
            lines.append(f"output logic          {rp}_tready")
            lines.append(f"input  logic          {rp}_tlast")
        else:
            lines.append(f"output logic [{dw-1}:0] {rp}_tdata")
            lines.append(f"output logic          {rp}_tvalid")
            lines.append(f"input  logic          {rp}_tready")
            lines.append(f"output logic          {rp}_tlast")
    elif iface.protocol == Protocol.AXI4:
        dw = iface.data_width or 256
        aw = iface.addr_width or 64
        rp = iface.rtl_port
        sw = dw // 8
        lines.append(f"// {iface_name}: AXI4 Master")
        # AR channel
        lines.append(f"output logic [{aw-1}:0] {rp}_araddr")
        lines.append(f"output logic [7:0]          {rp}_arlen")
        lines.append(f"output logic [2:0]          {rp}_arsize")
        lines.append(f"output logic [1:0]          {rp}_arburst")
        lines.append(f"output logic                {rp}_arvalid")
        lines.append(f"input  logic                {rp}_arready")
        # R channel
        lines.append(f"input  logic [{dw-1}:0] {rp}_rdata")
        lines.append(f"input  logic [1:0]          {rp}_rresp")
        lines.append(f"input  logic                {rp}_rlast")
        lines.append(f"input  logic                {rp}_rvalid")
        lines.append(f"output logic                {rp}_rready")
        # AW channel
        lines.append(f"output logic [{aw-1}:0] {rp}_awaddr")
        lines.append(f"output logic [7:0]          {rp}_awlen")
        lines.append(f"output logic [2:0]          {rp}_awsize")
        lines.append(f"output logic [1:0]          {rp}_awburst")
        lines.append(f"output logic                {rp}_awvalid")
        lines.append(f"input  logic                {rp}_awready")
        # W channel
        lines.append(f"output logic [{dw-1}:0] {rp}_wdata")
        lines.append(f"output logic [{sw-1}:0] {rp}_wstrb")
        lines.append(f"output logic                {rp}_wlast")
        lines.append(f"output logic                {rp}_wvalid")
        lines.append(f"input  logic                {rp}_wready")
        # B channel
        lines.append(f"input  logic [1:0]          {rp}_bresp")
        lines.append(f"input  logic                {rp}_bvalid")
        lines.append(f"output logic                {rp}_bready")
    return lines


def _generate_sub_kernel_instance(
    sk: dict,
    top_spec: KernelSpec,
    bindings: dict,
    internal_wires: list[dict],
) -> list[str]:
    """Generate instance lines for one sub-kernel wrapper."""
    from vten.kernel.composite import Internal

    sub_name = sk["name"]
    mod_name = sk["module_name"]
    sub_spec = sk["spec"]
    binding = sk["binding"]
    lines = []

    lines.append(f"{mod_name} u_{sub_name} (")
    lines.append("    .clk(clk),")
    lines.append("    .rst_n(rst_n),")

    port_connections = []
    for sub_iface_name, mapping in binding.interface_map.items():
        sub_iface = sub_spec.interfaces.get(sub_iface_name)
        if sub_iface is None:
            continue

        if isinstance(mapping, Internal):
            # Find the internal wire for this connection
            wire = _find_internal_wire(
                sub_name, sub_iface_name, internal_wires
            )
            if wire:
                port_connections.extend(
                    _wire_stream_to_internal(sub_iface, wire["name"])
                )
        elif isinstance(mapping, str):
            # External mapping — connect to top-level ports
            top_iface = top_spec.interfaces.get(mapping)
            if top_iface:
                port_connections.extend(
                    _wire_to_top(sub_iface, top_iface)
                )
        elif isinstance(mapping, tuple):
            # Independent ctrl — connect to top-level ctrl port
            top_name = mapping[0]
            top_iface = top_spec.interfaces.get(top_name)
            if top_iface:
                port_connections.extend(
                    _wire_to_top(sub_iface, top_iface)
                )

    for i, pc in enumerate(port_connections):
        comma = "," if i < len(port_connections) - 1 else ""
        lines.append(f"    {pc}{comma}")

    lines.append(");")
    return lines


def _find_internal_wire(
    sub_name: str, sub_iface_name: str, internal_wires: list[dict]
) -> dict | None:
    """Find the internal wire connected to this sub-kernel interface."""
    for wire in internal_wires:
        if (wire["source_sub"] == sub_name and
                wire["source_iface"] == sub_iface_name):
            return wire
        if (wire["dest_sub"] == sub_name and
                wire["dest_iface"] == sub_iface_name):
            return wire
    return None


def _wire_stream_to_internal(
    sub_iface: InterfaceSpec, wire_name: str
) -> list[str]:
    """Wire a sub-kernel stream port to an internal wire."""
    rp = sub_iface.rtl_port
    conns = []
    if rp.startswith("m_"):
        # Master output → internal wire (source)
        conns.append(f".{rp}_tdata({wire_name}_tdata)")
        conns.append(f".{rp}_tvalid({wire_name}_tvalid)")
        conns.append(f".{rp}_tready({wire_name}_tready)")
        conns.append(f".{rp}_tlast({wire_name}_tlast)")
    else:
        # Slave input ← internal wire (sink)
        conns.append(f".{rp}_tdata({wire_name}_tdata)")
        conns.append(f".{rp}_tvalid({wire_name}_tvalid)")
        conns.append(f".{rp}_tready({wire_name}_tready)")
        conns.append(f".{rp}_tlast({wire_name}_tlast)")
    return conns


def _wire_to_top(
    sub_iface: InterfaceSpec, top_iface: InterfaceSpec
) -> list[str]:
    """Wire a sub-kernel port to the corresponding top-level port."""
    sub_rp = sub_iface.rtl_port
    top_rp = top_iface.rtl_port
    conns = []

    if sub_iface.protocol == Protocol.AXI4L:
        for sig in ["awaddr", "awvalid", "awready", "wdata", "wstrb",
                     "wvalid", "wready", "bresp", "bvalid", "bready",
                     "araddr", "arvalid", "arready", "rdata", "rresp",
                     "rvalid", "rready"]:
            conns.append(f".{sub_rp}_{sig}({top_rp}_{sig})")
    elif sub_iface.protocol == Protocol.AXI4S:
        for sig in ["tdata", "tvalid", "tready", "tlast"]:
            conns.append(f".{sub_rp}_{sig}({top_rp}_{sig})")
    elif sub_iface.protocol == Protocol.AXI4:
        for sig in ["araddr", "arlen", "arsize", "arburst", "arvalid", "arready",
                     "rdata", "rresp", "rlast", "rvalid", "rready",
                     "awaddr", "awlen", "awsize", "awburst", "awvalid", "awready",
                     "wdata", "wstrb", "wlast", "wvalid", "wready",
                     "bresp", "bvalid", "bready"]:
            conns.append(f".{sub_rp}_{sig}({top_rp}_{sig})")
    return conns
