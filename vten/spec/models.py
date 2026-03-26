"""KernelSpec dataclass models.

Spec reference: 00_data_models.md §5, 03_kernel_spec_schema.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vten.errors import SpecValidationError, ValidationError


# ── Enums (from 00_data_models.md §1) ──


class Protocol(Enum):
    AXI4S = "axi4_stream"
    AXI4 = "axi4"
    AXI4L = "axi4_lite"


class Role(Enum):
    MASTER = "master"
    SLAVE = "slave"


class Direction(Enum):
    HOST_TO_DEV = "host_to_dev"
    DEV_TO_HOST = "dev_to_host"
    BIDIRECTIONAL = "bidirectional"


# ── OpCode (§1.4 — SHM IR Commands) ──


class OpCode(Enum):
    LOAD = 1
    PUSH = 2
    PULL = 3
    STORE = 4
    WRITE_REG = 5
    READ_REG = 6
    POLL_REG = 7
    BARRIER = 8
    COMPARE = 9


# ── OpKind (§1.5 — Record Phase Operations) ──


class OpKind(Enum):
    LOAD_TENSOR = "load_tensor"
    STORE_TENSOR = "store_tensor"
    PUSH_TENSOR = "push_tensor"
    PULL_TENSOR = "pull_tensor"
    WRITE_REGISTER = "write_register"
    READ_REGISTER = "read_register"
    POLL_REGISTER = "poll_register"
    CONFIGURE = "configure"
    BARRIER = "barrier"
    SEND_TENSOR = "send_tensor"
    RECV_TENSOR = "recv_tensor"


# ── MappingType (§1.6) ──


class MappingType(Enum):
    EXTERNAL = "external"
    EXTERNAL_BANK = "external_bank"
    INTERNAL = "internal"
    INTERNAL_PROBE = "internal_probe"


# ── CommandStatus (§1.7) ──


class CommandStatus(Enum):
    PENDING = 0
    ISSUED = 1
    ACTIVE = 2
    COMMITTED = 3
    ERROR = 4


# ── Packing (§5.1) ──


@dataclass
class CustomField:
    name: str
    bits: tuple[int, int]  # (lo_bit, hi_bit) inclusive


@dataclass
class PackingScheme:
    element_width: int
    elements_per_beat: int
    bit_order: str = "lsb_first"
    alignment: str = "packed"
    byte_order: str = "little"
    mode: str = "standard"
    custom_fields: list[CustomField] | None = None

    @property
    def bus_width(self) -> int:
        if self.mode == "custom" and self.custom_fields:
            return max(f.bits[1] for f in self.custom_fields) + 1
        if self.alignment == "packed":
            return self.element_width * self.elements_per_beat
        else:
            elem_bytes = (self.element_width + 7) // 8
            return elem_bytes * 8 * self.elements_per_beat

    def validate_custom_fields(self) -> None:
        if self.mode != "custom" or not self.custom_fields:
            return
        occupied: dict[int, str] = {}
        for cf in self.custom_fields:
            lo, hi = cf.bits
            for bit in range(lo, hi + 1):
                if bit in occupied:
                    raise ValidationError(
                        f"overlap: field '{cf.name}' bits {cf.bits} "
                        f"conflicts with '{occupied[bit]}' at bit {bit}."
                    )
                occupied[bit] = cf.name


# ── Split (§5.2) ──


@dataclass
class PortDef:
    name: str
    base_addr: int


@dataclass
class InterleaveSpec:
    unit: int


@dataclass
class SplitSpec:
    mode: str
    ports: list[PortDef]
    interleave: InterleaveSpec | None = None


# ── ArraySpec (spec 12) ──


@dataclass
class ArraySpec:
    dimensions: list[int]
    flat_name_pattern: str | None = None  # auto: {name}_{i} or {name}_{i}_{j}

    @property
    def total_elements(self) -> int:
        result = 1
        for d in self.dimensions:
            result *= d
        return result

    def flat_names(self, base_name: str) -> list[str]:
        """Generate flat names for all array elements in index order.

        Examples:
            ArraySpec([3]).flat_names("psum") -> ["psum_0", "psum_1", "psum_2"]
            ArraySpec([2,2], "wgt_{i}_{j}").flat_names("wgt")
                -> ["wgt_0_0", "wgt_0_1", "wgt_1_0", "wgt_1_1"]
        """
        from itertools import product as iterproduct

        pattern = self.flat_name_pattern
        if not pattern:
            var_names = "ijklmn"
            pattern = base_name + "".join(
                f"_{{{var_names[d]}}}" for d in range(len(self.dimensions))
            )

        var_names = "ijklmn"
        ranges = [range(d) for d in self.dimensions]
        names = []
        for indices in iterproduct(*ranges):
            idx_vars = {var_names[vi]: val for vi, val in enumerate(indices)}
            names.append(pattern.format(**idx_vars))
        return names


# ── AutoBind (§5.3) ──


@dataclass
class AutoBindSpec:
    tensor: str | None = None
    value: str | None = None
    bits: str | None = None
    param: str | None = None
    expr: str | None = None


# ── Register (§5.4) ──


@dataclass
class RegisterSpec:
    name: str
    offset: int
    fields: dict[str, str] | None = None
    auto_bind: AutoBindSpec | None = None
    interface_name: str = ""
    access: str = "rw"  # rw | ro | wo | w1c
    pulse: bool = False  # 1-cycle pulse (only with access=rw)
    reset_value: int = 0
    source: str = "software"  # "software" | "hardware"

    @property
    def width(self) -> int:
        """Register width in bits, inferred from fields or default 32."""
        if self.fields:
            max_bit = 0
            for bit_range in self.fields.values():
                hi, _lo = bit_range.split(":")
                max_bit = max(max_bit, int(hi))
            return max_bit + 1
        return 32


# ── MemoryRegion (§5.5) ──


@dataclass
class MemoryRegion:
    name: str
    base: int
    size: int
    alignment: int = 4096


# ── RegisterBankSpec (§5.6) ──


@dataclass
class RegisterBankSpec:
    name: str
    base_offset: int


# ── XrtInterfaceConfig (08_backend_abstraction.md §6.5) ──


@dataclass
class XrtInterfaceConfig:
    """XRT-specific interface configuration for FPGA deployment."""

    arg_index: int | None = None
    arg_name: str | None = None
    memory_bank: str | None = None
    ip_name: str | None = None
    memory_bank_index: int | None = None


# ── InterfaceSpec (§5.7) ──


@dataclass
class InterfaceSpec:
    name: str
    rtl_port: str
    protocol: Protocol
    data_width: int | None = None
    addr_width: int | None = None
    memory_region: str | None = None
    tensor: str | None = None
    tensors: list[str] | None = None
    packing: PackingScheme | None = None
    split: dict | SplitSpec | None = None
    registers: list[RegisterSpec] | None = None
    register_banks: list[RegisterBankSpec] | None = None
    generate_controller: bool = False
    user_register_base: int = 0x14  # Vitis reserved: 0x00-0x13
    array: ArraySpec | None = None
    role: str | None = None  # "master" | "slave"
    xrt: XrtInterfaceConfig | None = None

    @property
    def ext_port(self) -> str:
        """Vitis-compatible external port name prefix.

        Computes a standardized port name from protocol and interface name:
          AXI4-Lite  → s_axi_{name}
          AXI4 master → m_axi_{name}
          AXI4 slave  → s_axi_{name}
          AXI4-Stream master → m_axis_{name}
          AXI4-Stream slave  → s_axis_{name}
        """
        _role = self.role or (
            "master" if self.rtl_port.startswith("m_") else "slave"
        )
        if self.protocol == Protocol.AXI4L:
            return f"s_axi_{self.name}"
        elif self.protocol == Protocol.AXI4S:
            prefix = "m_axis" if _role == "master" else "s_axis"
            return f"{prefix}_{self.name}"
        elif self.protocol == Protocol.AXI4:
            prefix = "m_axi" if _role == "master" else "s_axi"
            return f"{prefix}_{self.name}"
        return self.rtl_port


# ── KernelSpec (§5.8) ──


@dataclass
class KernelSpec:
    kernel_name: str
    rtl_top: str
    parameters: dict[str, str | int] = field(default_factory=dict)
    memory_regions: dict[str, MemoryRegion] = field(default_factory=dict)
    interfaces: dict[str, InterfaceSpec] = field(default_factory=dict)
    clock_name: str = "clk"
    reset_name: str = "rst_n"
    reset_active_low: bool = True

    def get_interface(self, name: str) -> InterfaceSpec:
        if name not in self.interfaces:
            raise KeyError(f"Interface '{name}' not found")
        return self.interfaces[name]

    def get_registers(self, interface_name: str) -> list[RegisterSpec]:
        iface = self.get_interface(interface_name)
        return iface.registers or []

    def interface_names(self) -> list[str]:
        return list(self.interfaces.keys())

    def expanded_interface_names(self) -> list[str]:
        """Interface names with array interfaces expanded to flat elements.

        Non-array interfaces return their name as-is.
        Array interfaces expand to flat element names in index order.
        Used by both IR lowering and codegen to ensure consistent ID assignment.
        """
        result: list[str] = []
        for name, iface in self.interfaces.items():
            if iface.array:
                result.extend(iface.array.flat_names(name))
            else:
                result.append(name)
        return result

    def resolve_flat_interface(self, flat_name: str) -> tuple[InterfaceSpec, str]:
        """Given a flat element name, return (parent InterfaceSpec, logical name).

        For non-array interfaces, flat_name == logical name.
        For array elements like 'wgt_0_1', returns (wgt InterfaceSpec, 'wgt').
        """
        # Direct match first
        if flat_name in self.interfaces:
            return self.interfaces[flat_name], flat_name
        # Search array interfaces
        for name, iface in self.interfaces.items():
            if iface.array and flat_name in iface.array.flat_names(name):
                return iface, name
        raise KeyError(f"Interface '{flat_name}' not found")

    def get_bank_offset(self, interface_name: str, bank_name: str) -> int:
        iface = self.get_interface(interface_name)
        if iface.register_banks:
            for bank in iface.register_banks:
                if bank.name == bank_name:
                    return bank.base_offset
        raise ValueError(
            f"Bank '{bank_name}' not found in interface '{interface_name}'"
        )
