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

    def get_bank_offset(self, interface_name: str, bank_name: str) -> int:
        iface = self.get_interface(interface_name)
        if iface.register_banks:
            for bank in iface.register_banks:
                if bank.name == bank_name:
                    return bank.base_offset
        raise ValueError(
            f"Bank '{bank_name}' not found in interface '{interface_name}'"
        )
