"""kernel_spec.yaml parser.

Spec reference: 03_kernel_spec_schema.md §14
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from vten.errors import BankOverlapError, SpecValidationError
from vten.spec.models import (
    ArraySpec,
    AutoBindSpec,
    CustomField,
    InterfaceSpec,
    InterleaveSpec,
    KernelSpec,
    MemoryRegion,
    PackingScheme,
    PortDef,
    Protocol,
    RegisterBankSpec,
    RegisterSpec,
    SplitSpec,
)


def parse_kernel_spec(yaml_path: str | Path) -> KernelSpec:
    """Parse kernel_spec.yaml → KernelSpec dataclass."""
    return load_kernel_spec(yaml_path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. override wins for scalar/list values."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_includes(raw: dict, spec_dir: Path) -> dict:
    """Process 'include' directive: deep-merge included files as defaults."""
    includes = raw.pop("include", None)
    if not includes:
        return raw
    if isinstance(includes, str):
        includes = [includes]

    base: dict = {}
    for inc_path_str in includes:
        inc_path = (spec_dir / inc_path_str).resolve()
        if not inc_path.exists():
            raise SpecValidationError(
                f"Include file not found: {inc_path} "
                f"(referenced from {spec_dir})"
            )
        with open(inc_path) as f:
            inc_data = yaml.safe_load(f) or {}
        base = _deep_merge(base, inc_data)

    # Merge: base (from includes) is defaults, raw (the spec) overrides
    return _deep_merge(base, raw)


def load_kernel_spec(yaml_path: str | Path) -> KernelSpec:
    """Parse kernel_spec.yaml → KernelSpec dataclass."""
    path = Path(yaml_path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_includes(raw, path.parent)
    _validate_top_level(raw, path)

    parameters = raw.get("parameters", {}) or {}
    build_params = raw.get("build_params", {}) or {}
    memory_regions = {
        name: _parse_memory_region(name, spec)
        for name, spec in (raw.get("memory_regions") or {}).items()
    }
    interfaces = {
        name: _parse_interface(name, spec, memory_regions, spec_dir=path.parent)
        for name, spec in raw["interfaces"].items()
    }

    # Clock/reset config (optional, defaults in KernelSpec)
    clock_cfg = raw.get("clock", {}) or {}
    reset_cfg = raw.get("reset", {}) or {}

    spec = KernelSpec(
        kernel_name=raw["kernel"],
        rtl_top=raw["rtl_top"],
        parameters=parameters,
        build_params=build_params,
        memory_regions=memory_regions,
        interfaces=interfaces,
        clock_name=clock_cfg.get("name", "clk"),
        reset_name=reset_cfg.get("name", "rst_n"),
        reset_active_low=reset_cfg.get("active_low", True),
    )

    # Post-parse validation
    for iface in interfaces.values():
        _validate_packing(iface)
        _validate_register_banks(iface)

    return spec


def _validate_top_level(raw: dict, path: Path) -> None:
    for key in ("kernel", "rtl_top", "interfaces"):
        if key not in raw:
            raise SpecValidationError(
                f"Missing required top-level field '{key}' in {path}"
            )
    if not raw["interfaces"]:
        raise SpecValidationError(
            f"At least one interface required in {path}"
        )


def _parse_memory_region(name: str, spec: dict) -> MemoryRegion:
    return MemoryRegion(
        name=name,
        base=spec["base"],
        size=spec["size"],
        alignment=spec.get("alignment", 4096),
    )


def _parse_interface(
    name: str,
    spec: dict,
    memory_regions: dict[str, MemoryRegion],
    spec_dir: Path | None = None,
) -> InterfaceSpec:
    if "protocol" not in spec:
        raise SpecValidationError(
            f"Interface '{name}': missing 'protocol'"
        )

    protocol = _parse_protocol(spec["protocol"], name)
    role = spec.get("role")

    # rtl_port: explicit or auto-generated from protocol + role + name
    if "rtl_port" in spec:
        rtl_port = spec["rtl_port"]
    elif role:
        rtl_port = _auto_rtl_port(protocol, role, name)
    else:
        raise SpecValidationError(
            f"Interface '{name}': either 'rtl_port' or 'role' must be specified"
        )

    # Parse sub-structures
    packing = _parse_packing(spec.get("packing")) if "packing" in spec else None
    split = spec.get("split")  # Store as raw dict for dict-style access
    user_register_base = spec.get("user_register_base", 0x14)

    # register_include: load external register definitions, prepend before local
    reg_includes = spec.get("register_include") or []
    if isinstance(reg_includes, str):
        reg_includes = [reg_includes]
    included_regs: list[dict] = []
    for inc_path_str in reg_includes:
        if spec_dir is None:
            raise SpecValidationError(
                f"Interface '{name}': register_include requires spec_dir"
            )
        inc_path = (spec_dir / inc_path_str).resolve()
        if not inc_path.exists():
            raise SpecValidationError(
                f"Interface '{name}': register_include file not found: {inc_path}"
            )
        with open(inc_path) as f:
            inc_regs = yaml.safe_load(f)
        if not isinstance(inc_regs, list):
            raise SpecValidationError(
                f"Interface '{name}': register_include file must contain a YAML list"
            )
        included_regs.extend(inc_regs)

    local_regs = spec.get("registers") or []
    combined_regs = included_regs + local_regs
    registers = _parse_registers(combined_regs, name, user_register_base) if combined_regs else None
    register_banks = _parse_register_banks(spec.get("register_banks")) if "register_banks" in spec else None

    # Determine data_width and addr_width defaults
    data_width = spec.get("data_width")
    addr_width = spec.get("addr_width")
    if addr_width is None:
        if protocol == Protocol.AXI4:
            addr_width = 64
        elif protocol == Protocol.AXI4L:
            addr_width = 12  # 4KB range — standard for Xilinx AXI-Lite IP

    # tensor / tensors handling
    tensor = spec.get("tensor")
    tensors = spec.get("tensors")

    # Mutual exclusivity validation
    if tensor is not None and tensors is not None:
        raise SpecValidationError(
            f"Interface '{name}': 'tensor' and 'tensors' are mutually exclusive"
        )

    # Validate memory_region reference for AXI4
    memory_region = spec.get("memory_region")
    if protocol == Protocol.AXI4 and memory_region:
        if memory_region not in memory_regions:
            raise SpecValidationError(
                f"Interface '{name}': memory_region '{memory_region}' "
                f"not defined in memory_regions"
            )

    generate_controller = spec.get("generate_controller", False)
    if generate_controller and protocol != Protocol.AXI4L:
        raise SpecValidationError(
            f"Interface '{name}': generate_controller is only valid "
            f"for axi4_lite interfaces"
        )

    # Array spec (spec 12)
    array = _parse_array(spec.get("array"), name) if "array" in spec else None

    # XRT configuration (08_backend_abstraction.md §6.5)
    xrt_config = None
    if "xrt" in spec:
        from vten.spec.models import XrtInterfaceConfig

        xrt_raw = spec["xrt"]
        xrt_config = XrtInterfaceConfig(
            arg_index=xrt_raw.get("arg_index"),
            arg_name=xrt_raw.get("arg_name"),
            memory_bank=xrt_raw.get("memory_bank"),
            ip_name=xrt_raw.get("ip_name"),
            memory_bank_index=xrt_raw.get("memory_bank_index"),
        )

    return InterfaceSpec(
        name=name,
        rtl_port=rtl_port,
        protocol=protocol,
        data_width=data_width,
        addr_width=addr_width,
        memory_region=memory_region,
        tensor=tensor,
        tensors=tensors,
        packing=packing,
        split=split,
        registers=registers,
        register_banks=register_banks,
        generate_controller=generate_controller,
        user_register_base=user_register_base,
        array=array,
        role=role,
        xrt=xrt_config,
    )


_RTL_PORT_PREFIX = {
    (Protocol.AXI4S, "master"): "m_axis_",
    (Protocol.AXI4S, "slave"): "s_axis_",
    (Protocol.AXI4, "master"): "m_axi_",
    (Protocol.AXI4, "slave"): "s_axi_",
    (Protocol.AXI4L, "slave"): "s_axilite_",
    (Protocol.AXI4L, "master"): "m_axilite_",
}


def _auto_rtl_port(protocol: Protocol, role: str, name: str) -> str:
    """Auto-generate rtl_port from protocol + role + name (Vitis naming)."""
    key = (protocol, role)
    if key not in _RTL_PORT_PREFIX:
        raise SpecValidationError(
            f"Interface '{name}': no default rtl_port prefix for "
            f"protocol={protocol.value}, role={role}"
        )
    return _RTL_PORT_PREFIX[key] + name


def _parse_array(raw: dict, iface_name: str) -> ArraySpec:
    """Parse array spec from interface definition."""
    dims = raw.get("dimensions")
    if not dims or not isinstance(dims, list):
        raise SpecValidationError(
            f"Interface '{iface_name}': array.dimensions must be a non-empty list"
        )
    for d in dims:
        if not isinstance(d, int) or d <= 0:
            raise SpecValidationError(
                f"Interface '{iface_name}': array.dimensions values must be positive integers"
            )
    # flat_name_pattern: optional — auto-generated from name + dimensions
    # e.g. 1D: "{name}_{i}", 2D: "{name}_{i}_{j}"
    pattern = raw.get("flat_name_pattern")  # None if omitted
    interleave = None
    if "interleave" in raw:
        from vten.spec.models import InterleaveSpec
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return ArraySpec(dimensions=dims, flat_name_pattern=pattern, interleave=interleave)


def _parse_protocol(value: str, iface_name: str) -> Protocol:
    for p in Protocol:
        if p.value == value:
            return p
    raise SpecValidationError(
        f"Interface '{iface_name}': unknown protocol '{value}'"
    )


def _parse_packing(raw: dict) -> PackingScheme:
    mode = raw.get("mode", "standard")
    if mode == "custom":
        fields_raw = raw.get("fields", [])
        custom_fields = [
            CustomField(name=f["name"], bits=tuple(f["bits"]))
            for f in fields_raw
        ]
        scheme = PackingScheme(
            element_width=0,
            elements_per_beat=0,
            mode="custom",
            custom_fields=custom_fields,
        )
        scheme.validate_custom_fields()
        return scheme

    scheme = PackingScheme(
        element_width=raw["element_width"],
        elements_per_beat=raw["elements_per_beat"],
        bit_order=raw.get("bit_order", "lsb_first"),
        alignment=raw.get("alignment", "packed"),
        byte_order=raw.get("byte_order", "little"),
        mode="standard",
    )
    if "bus_width" in raw:
        scheme._explicit_bus_width = raw["bus_width"]
    return scheme


def _parse_split(raw: dict) -> SplitSpec:
    ports = [
        PortDef(name=p["name"], base_addr=p["base_addr"])
        for p in raw["ports"]
    ]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)


def _parse_registers(
    raw: list, interface_name: str, user_register_base: int = 0x14
) -> list[RegisterSpec]:
    registers = []
    next_offset = user_register_base
    for r in raw:
        # width shorthand: { name: foo, width: 10 } → fields: { foo: "9:0" }
        if "width" in r and "fields" not in r:
            w = r["width"]
            r = dict(r)  # avoid mutating original
            r["fields"] = {r["name"]: f"{w - 1}:0"}

        auto_bind = None
        if "auto_bind" in r:
            ab = r["auto_bind"]
            auto_bind = AutoBindSpec(
                tensor=ab.get("tensor"),
                value=ab.get("value"),
                bits=ab.get("bits"),
                param=ab.get("param"),
                expr=ab.get("expr"),
                offset=ab.get("offset"),
            )
        access = r.get("access", "rw")
        if access not in ("rw", "ro", "wo", "w1c"):
            raise SpecValidationError(
                f"Register '{r['name']}': invalid access '{access}'"
            )
        pulse = r.get("pulse", False)
        if pulse and access not in ("rw", "wo"):
            raise SpecValidationError(
                f"Register '{r['name']}': pulse is only valid with "
                f"access='rw' or 'wo'"
            )
        # Offset: explicit or auto-assigned from user_register_base
        if "offset" in r:
            offset = r["offset"]
            next_offset = offset + 4
        else:
            offset = next_offset
            next_offset += 4

        registers.append(
            RegisterSpec(
                name=r["name"],
                offset=offset,
                fields=r.get("fields"),
                auto_bind=auto_bind,
                interface_name=interface_name,
                access=access,
                pulse=pulse,
                reset_value=r.get("reset_value", 0),
            )
        )
    return registers


def _parse_register_banks(raw: dict) -> list[RegisterBankSpec]:
    banks = []
    for name, spec in raw.items():
        banks.append(
            RegisterBankSpec(name=name, base_offset=spec["base_offset"])
        )
    return banks


def _validate_register_banks(iface: InterfaceSpec) -> None:
    """V3: Register bank base_offsets must not overlap."""
    if not iface.register_banks or len(iface.register_banks) < 2:
        return

    seen: dict[int, str] = {}
    for bank in iface.register_banks:
        if bank.base_offset in seen:
            raise BankOverlapError(
                f"Interface '{iface.name}': register banks "
                f"'{seen[bank.base_offset]}' and '{bank.name}' "
                f"have the same base_offset 0x{bank.base_offset:X}"
            )
        seen[bank.base_offset] = bank.name


def _validate_packing(iface: InterfaceSpec) -> None:
    if iface.packing is None:
        return

    iface.packing.validate_custom_fields()
    bus_width = iface.packing.bus_width

    if iface.protocol == Protocol.AXI4:
        if iface.data_width is None:
            raise SpecValidationError(
                f"AXI4 interface '{iface.name}' requires explicit data_width"
            )
        if bus_width > iface.data_width:
            raise SpecValidationError(
                f"Interface '{iface.name}': packing bus_width ({bus_width}) "
                f"exceeds data_width ({iface.data_width}). "
                f"Reduce elements_per_beat or element_width."
            )
        if bus_width < iface.data_width:
            warnings.warn(
                f"Interface '{iface.name}': packing bus_width ({bus_width}) "
                f"< data_width ({iface.data_width}). "
                f"Upper {iface.data_width - bus_width} bits will be zero-padded."
            )
