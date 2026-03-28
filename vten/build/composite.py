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


# ── Probe Info Extraction ──


def extract_probe_bfm_info(
    composite_cls: type,
    project: Path,
) -> list[dict]:
    """Extract probe BFM info from composite class for codegen.

    Returns a list of dicts, one per probed internal connection:
      - probe_index: sequential index (0, 1, ...)
      - wire_name: internal wire name in composite wrapper (e.g. "internal_0")
      - data_width: bus width of the internal wire
      - connection_index: index into connections list
    """
    from vten.kernel.composite import Internal

    bindings = dict(composite_cls._sub_kernel_bindings)
    connections = list(composite_cls._connections)

    # Build set of (sub_name, sub_iface_name) that are Internal(probe=True)
    probed_interfaces: set[tuple[str, str]] = set()
    for sub_name, binding in bindings.items():
        for sub_iface_name, mapping in binding.interface_map.items():
            if isinstance(mapping, Internal) and mapping.probe:
                probed_interfaces.add((sub_name, sub_iface_name))

    probe_bfms = []
    probe_index = 0
    for conn_idx, conn in enumerate(connections):
        # A connection is probed if either end's interface is Internal(probe=True)
        src_key = (conn.source_sub, conn.source_interface)
        dst_iface = _find_dest_interface(conn, bindings)
        dst_key = (conn.dest_sub, dst_iface) if dst_iface else None

        is_probed = src_key in probed_interfaces or (
            dst_key is not None and dst_key in probed_interfaces
        )
        if not is_probed:
            continue

        # Get data width from source interface
        src_binding = bindings[conn.source_sub]
        src_spec = _load_sub_kernel_spec(src_binding.kernel_class, project)
        src_iface = src_spec.interfaces.get(conn.source_interface)
        data_w = 256
        if src_iface and src_iface.packing:
            data_w = src_iface.packing.bus_width

        wire_name = f"internal_{conn_idx}"
        probe_bfms.append({
            "probe_index": probe_index,
            "wire_name": wire_name,
            "data_width": data_w,
            "connection_index": conn_idx,
        })
        probe_index += 1

    return probe_bfms


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

    # Build internal wire list from connections.
    # Array interfaces expand into one wire per element.
    internal_wires = []
    for idx, conn in enumerate(connections):
        wire_base = f"internal_{idx}"
        src_sub = conn.source_sub
        src_binding = bindings[src_sub]
        src_spec = _load_sub_kernel_spec(src_binding.kernel_class, project)
        src_iface = src_spec.interfaces.get(conn.source_interface)
        data_w = 256
        if src_iface and src_iface.packing:
            data_w = src_iface.packing.bus_width
        protocol = src_iface.protocol if src_iface else Protocol.AXI4S
        addr_w = src_iface.addr_width or 64 if src_iface else 64
        dest_iface_name = _find_dest_interface(conn, bindings)

        if src_iface and src_iface.array:
            flat_src = src_iface.array.flat_names(conn.source_interface)
            # Resolve dest flat names
            dest_binding = bindings[conn.dest_sub]
            dest_spec = _load_sub_kernel_spec(dest_binding.kernel_class, project)
            dest_iface = dest_spec.interfaces.get(dest_iface_name) if dest_iface_name else None
            flat_dst = dest_iface.array.flat_names(dest_iface_name) if (dest_iface and dest_iface.array) else flat_src
            for ei, (sf, df) in enumerate(zip(flat_src, flat_dst)):
                suffix = sf[len(conn.source_interface):]
                internal_wires.append({
                    "name": f"{wire_base}{suffix}",
                    "data_w": data_w,
                    "protocol": protocol,
                    "addr_w": addr_w,
                    "source_sub": src_sub,
                    "source_iface": conn.source_interface,
                    "source_flat": sf,
                    "dest_sub": conn.dest_sub,
                    "dest_iface": dest_iface_name,
                    "dest_flat": df,
                    "is_array_element": True,
                })
        else:
            internal_wires.append({
                "name": wire_base,
                "data_w": data_w,
                "protocol": protocol,
                "addr_w": addr_w,
                "source_sub": src_sub,
                "source_iface": conn.source_interface,
                "dest_sub": conn.dest_sub,
                "dest_iface": dest_iface_name,
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
            lines.extend(
                f"    {l}" for l in _declare_internal_wire(wire)
            )
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
    """Generate port declaration lines for a top-level interface.

    Uses ext_port (Vitis-compatible name) so composite wrapper ports match
    the tb_top wire naming convention (s_axi_{name}, s_axis_{name}, etc.).
    """
    lines = []
    if iface.protocol == Protocol.AXI4L:
        prefix = iface.ext_port
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
        rp = iface.ext_port
        # Array expansion: one port set per element
        if iface.array:
            flat_names = iface.array.flat_names(iface_name)
            lines.append(f"// {iface_name}: AXI4-Stream array[{iface.array.total_elements}]")
            for fn in flat_names:
                suffix = fn[len(iface_name):]
                erp = f"{rp}{suffix}"
                if rp.startswith("s_"):
                    lines.extend([
                        f"input  logic [{dw-1}:0] {erp}_tdata",
                        f"input  logic          {erp}_tvalid",
                        f"output logic          {erp}_tready",
                        f"input  logic          {erp}_tlast",
                    ])
                else:
                    lines.extend([
                        f"output logic [{dw-1}:0] {erp}_tdata",
                        f"output logic          {erp}_tvalid",
                        f"input  logic          {erp}_tready",
                        f"output logic          {erp}_tlast",
                    ])
        else:
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
        rp = iface.ext_port
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

    # Sub-kernel wrappers with generate_controller use ap_clk/ap_aresetn;
    # plain cores use clk/rst_n.
    has_ctrl = any(
        iface.generate_controller
        for iface in sub_spec.interfaces.values()
    )
    lines.append(f"{mod_name} u_{sub_name} (")
    if has_ctrl:
        lines.append("    .ap_clk(clk),")
        lines.append("    .ap_aresetn(rst_n),")
    else:
        lines.append("    .clk(clk),")
        lines.append("    .rst_n(rst_n),")

    port_connections = []
    for sub_iface_name, mapping in binding.interface_map.items():
        sub_iface = sub_spec.interfaces.get(sub_iface_name)
        if sub_iface is None:
            continue

        if isinstance(mapping, Internal):
            wire_or_wires = _find_internal_wire(
                sub_name, sub_iface_name, internal_wires
            )
            if wire_or_wires is not None:
                if isinstance(wire_or_wires, list):
                    port_connections.extend(
                        _wire_array_to_internal(sub_iface, sub_name, wire_or_wires)
                    )
                else:
                    port_connections.extend(
                        _wire_to_internal(sub_iface, wire_or_wires)
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
) -> dict | list[dict] | None:
    """Find internal wire(s) for a sub-kernel interface.

    Returns list for array interfaces, single dict for scalar.
    """
    matches = []
    for wire in internal_wires:
        if (wire["source_sub"] == sub_name and
                wire["source_iface"] == sub_iface_name):
            matches.append(wire)
        elif (wire["dest_sub"] == sub_name and
                wire["dest_iface"] == sub_iface_name):
            matches.append(wire)
    if not matches:
        return None
    if len(matches) == 1 and not matches[0].get("is_array_element"):
        return matches[0]
    return matches


def _declare_internal_wire(wire: dict) -> list[str]:
    """Declare internal wire signals based on protocol."""
    from vten.errors import BuildError

    wn = wire["name"]
    dw = wire["data_w"]
    protocol = wire.get("protocol", Protocol.AXI4S)

    if protocol == Protocol.AXI4S:
        return [
            f"logic [{dw-1}:0] {wn}_tdata;",
            f"logic           {wn}_tvalid;",
            f"logic           {wn}_tready;",
            f"logic           {wn}_tlast;",
        ]
    elif protocol == Protocol.AXI4:
        aw = wire.get("addr_w", 64)
        sw = dw // 8
        return [
            f"// {wn}: AXI4 internal",
            # AR channel
            f"logic [{aw-1}:0] {wn}_araddr;",
            f"logic [7:0]          {wn}_arlen;",
            f"logic [2:0]          {wn}_arsize;",
            f"logic [1:0]          {wn}_arburst;",
            f"logic                {wn}_arvalid;",
            f"logic                {wn}_arready;",
            # R channel
            f"logic [{dw-1}:0] {wn}_rdata;",
            f"logic [1:0]          {wn}_rresp;",
            f"logic                {wn}_rlast;",
            f"logic                {wn}_rvalid;",
            f"logic                {wn}_rready;",
            # AW channel
            f"logic [{aw-1}:0] {wn}_awaddr;",
            f"logic [7:0]          {wn}_awlen;",
            f"logic [2:0]          {wn}_awsize;",
            f"logic [1:0]          {wn}_awburst;",
            f"logic                {wn}_awvalid;",
            f"logic                {wn}_awready;",
            # W channel
            f"logic [{dw-1}:0] {wn}_wdata;",
            f"logic [{sw-1}:0] {wn}_wstrb;",
            f"logic                {wn}_wlast;",
            f"logic                {wn}_wvalid;",
            f"logic                {wn}_wready;",
            # B channel
            f"logic [1:0]          {wn}_bresp;",
            f"logic                {wn}_bvalid;",
            f"logic                {wn}_bready;",
        ]
    elif protocol == Protocol.AXI4L:
        aw = wire.get("addr_w", 16)
        dw = wire.get("data_w", 32)
        sw = dw // 8
        return [
            f"// {wn}: AXI4-Lite internal",
            f"logic [{aw-1}:0] {wn}_awaddr;",
            f"logic           {wn}_awvalid;",
            f"logic           {wn}_awready;",
            f"logic [{dw-1}:0] {wn}_wdata;",
            f"logic [{sw-1}:0] {wn}_wstrb;",
            f"logic           {wn}_wvalid;",
            f"logic           {wn}_wready;",
            f"logic [1:0]    {wn}_bresp;",
            f"logic           {wn}_bvalid;",
            f"logic           {wn}_bready;",
            f"logic [{aw-1}:0] {wn}_araddr;",
            f"logic           {wn}_arvalid;",
            f"logic           {wn}_arready;",
            f"logic [{dw-1}:0] {wn}_rdata;",
            f"logic [1:0]    {wn}_rresp;",
            f"logic           {wn}_rvalid;",
            f"logic           {wn}_rready;",
        ]
    else:
        raise BuildError(
            f"Unsupported protocol for internal wire '{wn}': {protocol}"
        )


def _wire_array_to_internal(
    sub_iface: InterfaceSpec, sub_name: str, wires: list[dict]
) -> list[str]:
    """Wire array sub-kernel ports to per-element internal wires."""
    conns = []
    rp = sub_iface.ext_port
    base = sub_iface.name
    for wire in wires:
        wn = wire["name"]
        flat = wire.get("source_flat") if wire.get("source_sub") == sub_name else wire.get("dest_flat")
        suffix = flat[len(base):] if flat else ""
        for sig in ["tdata", "tvalid", "tready", "tlast"]:
            conns.append(f".{rp}{suffix}_{sig}({wn}_{sig})")
    return conns


def _wire_to_internal(
    sub_iface: InterfaceSpec, wire: dict
) -> list[str]:
    """Wire a sub-kernel port to an internal wire (protocol-aware dispatcher)."""
    from vten.errors import BuildError

    protocol = wire.get("protocol", Protocol.AXI4S)
    wire_name = wire["name"]
    rp = sub_iface.ext_port

    if protocol == Protocol.AXI4S:
        return [
            f".{rp}_tdata({wire_name}_tdata)",
            f".{rp}_tvalid({wire_name}_tvalid)",
            f".{rp}_tready({wire_name}_tready)",
            f".{rp}_tlast({wire_name}_tlast)",
        ]
    elif protocol == Protocol.AXI4:
        sigs = [
            "araddr", "arlen", "arsize", "arburst", "arvalid", "arready",
            "rdata", "rresp", "rlast", "rvalid", "rready",
            "awaddr", "awlen", "awsize", "awburst", "awvalid", "awready",
            "wdata", "wstrb", "wlast", "wvalid", "wready",
            "bresp", "bvalid", "bready",
        ]
        return [f".{rp}_{sig}({wire_name}_{sig})" for sig in sigs]
    elif protocol == Protocol.AXI4L:
        sigs = [
            "awaddr", "awvalid", "awready",
            "wdata", "wstrb", "wvalid", "wready",
            "bresp", "bvalid", "bready",
            "araddr", "arvalid", "arready",
            "rdata", "rresp", "rvalid", "rready",
        ]
        return [f".{rp}_{sig}({wire_name}_{sig})" for sig in sigs]
    else:
        raise BuildError(
            f"Unsupported protocol for internal connection: {protocol}"
        )


def _wire_to_top(
    sub_iface: InterfaceSpec, top_iface: InterfaceSpec
) -> list[str]:
    """Wire a sub-kernel port to the corresponding top-level port.

    Both sides use ext_port (Vitis-compatible naming):
    - sub_rp: sub-kernel wrapper port (e.g., "s_axi_ctrl")
    - top_rp: composite top-level port (e.g., "s_axi_scale_ctrl")
    """
    sub_rp = sub_iface.ext_port
    top_rp = top_iface.ext_port
    conns = []

    if sub_iface.protocol == Protocol.AXI4L:
        for sig in ["awaddr", "awvalid", "awready", "wdata", "wstrb",
                     "wvalid", "wready", "bresp", "bvalid", "bready",
                     "araddr", "arvalid", "arready", "rdata", "rresp",
                     "rvalid", "rready"]:
            conns.append(f".{sub_rp}_{sig}({top_rp}_{sig})")
    elif sub_iface.protocol == Protocol.AXI4S:
        if sub_iface.array:
            sb = sub_iface.name
            tb = top_iface.name
            for sf, tf in zip(
                sub_iface.array.flat_names(sb),
                top_iface.array.flat_names(tb) if top_iface.array else sub_iface.array.flat_names(sb),
            ):
                ss, ts = sf[len(sb):], tf[len(tb):]
                for sig in ["tdata", "tvalid", "tready", "tlast"]:
                    conns.append(f".{sub_rp}{ss}_{sig}({top_rp}{ts}_{sig})")
        else:
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
