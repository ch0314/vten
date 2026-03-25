"""kernel_spec.yaml parser.

Spec reference: 03_kernel_spec_schema.md §14
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from vten.errors import BankOverlapError, SpecValidationError
from vten.spec.models import (
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


def load_kernel_spec(yaml_path: str | Path) -> KernelSpec:
    """Parse kernel_spec.yaml → KernelSpec dataclass."""
    path = Path(yaml_path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_top_level(raw, path)

    parameters = raw.get("parameters", {}) or {}
    memory_regions = {
        name: _parse_memory_region(name, spec)
        for name, spec in (raw.get("memory_regions") or {}).items()
    }
    interfaces = {
        name: _parse_interface(name, spec, memory_regions)
        for name, spec in raw["interfaces"].items()
    }

    # Clock/reset config (optional, defaults in KernelSpec)
    clock_cfg = raw.get("clock", {}) or {}
    reset_cfg = raw.get("reset", {}) or {}

    spec = KernelSpec(
        kernel_name=raw["kernel"],
        rtl_top=raw["rtl_top"],
        parameters=parameters,
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
    name: str, spec: dict, memory_regions: dict[str, MemoryRegion]
) -> InterfaceSpec:
    if "rtl_port" not in spec:
        raise SpecValidationError(
            f"Interface '{name}': missing 'rtl_port'"
        )
    if "protocol" not in spec:
        raise SpecValidationError(
            f"Interface '{name}': missing 'protocol'"
        )

    protocol = _parse_protocol(spec["protocol"], name)

    # Parse sub-structures
    packing = _parse_packing(spec.get("packing")) if "packing" in spec else None
    split = spec.get("split")  # Store as raw dict for dict-style access
    registers = _parse_registers(spec.get("registers"), name) if "registers" in spec else None
    register_banks = _parse_register_banks(spec.get("register_banks")) if "register_banks" in spec else None

    # Determine data_width and addr_width defaults
    data_width = spec.get("data_width")
    addr_width = spec.get("addr_width")
    if addr_width is None:
        if protocol == Protocol.AXI4:
            addr_width = 64
        elif protocol == Protocol.AXI4L:
            addr_width = 32

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

    # XRT configuration (08_backend_abstraction.md §6.5)
    xrt_config = None
    if "xrt" in spec:
        from vten.spec.models import XrtInterfaceConfig

        xrt_raw = spec["xrt"]
        xrt_config = XrtInterfaceConfig(
            arg_index=xrt_raw.get("arg_index"),
            arg_name=xrt_raw.get("arg_name"),
            memory_bank=xrt_raw.get("memory_bank"),
        )

    return InterfaceSpec(
        name=name,
        rtl_port=spec["rtl_port"],
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
        xrt=xrt_config,
    )


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

    return PackingScheme(
        element_width=raw["element_width"],
        elements_per_beat=raw["elements_per_beat"],
        bit_order=raw.get("bit_order", "lsb_first"),
        alignment=raw.get("alignment", "packed"),
        byte_order=raw.get("byte_order", "little"),
        mode="standard",
    )


def _parse_split(raw: dict) -> SplitSpec:
    ports = [
        PortDef(name=p["name"], base_addr=p["base_addr"])
        for p in raw["ports"]
    ]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)


def _parse_registers(raw: list, interface_name: str) -> list[RegisterSpec]:
    registers = []
    for r in raw:
        auto_bind = None
        if "auto_bind" in r:
            ab = r["auto_bind"]
            auto_bind = AutoBindSpec(
                tensor=ab.get("tensor"),
                value=ab.get("value"),
                bits=ab.get("bits"),
                param=ab.get("param"),
                expr=ab.get("expr"),
            )
        access = r.get("access", "rw")
        if access not in ("rw", "ro", "wo", "w1c"):
            raise SpecValidationError(
                f"Register '{r['name']}': invalid access '{access}'"
            )
        pulse = r.get("pulse", False)
        if pulse and access != "rw":
            raise SpecValidationError(
                f"Register '{r['name']}': pulse is only valid with access='rw'"
            )
        registers.append(
            RegisterSpec(
                name=r["name"],
                offset=r["offset"],
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
