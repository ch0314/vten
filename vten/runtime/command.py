"""IR command and BFM configuration dataclasses.

Spec reference: 00_data_models.md §9–10
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vten.spec.models import (
    DEFAULT_DATA_WIDTH,
    OpCode,
    Protocol,
    Role,
)


@dataclass
class BFMConfig:
    """BFM instance configuration."""

    interface_name: str
    protocol: Protocol
    data_width: int = DEFAULT_DATA_WIDTH
    addr_width: int = 64
    role: str = "slave"
    address_ranges: list[tuple[int, int, int]] = field(default_factory=list)
    poll_interval: int = 1
    poll_timeout: int = 100000


@dataclass
class Command:
    """Single IR command. Packed to 64-byte SHM slot."""

    op: OpCode
    cmd_id: int
    interface_id: int = 0
    buffer_id: int = 0
    protocol: Protocol = Protocol.AXI4S
    phys_addr: int = 0
    size: int = 0
    role: Role = Role.MASTER
    dep: list[int] = field(default_factory=list)
    commit_dep: list[int] = field(default_factory=list)
    reg_offset: int = 0
    reg_value: int = 0
    reg_mask: int = 0
    reg_expected: int = 0
    probe: bool = False
    golden_buf: int = 0
    sync: bool = False
    port: str = ""


def determine_role(protocol: Protocol, opcode: OpCode) -> Role:
    """Determine BFM role from protocol and opcode."""
    if protocol == Protocol.AXI4L:
        return Role.MASTER
    if protocol == Protocol.AXI4S:
        return Role.MASTER if opcode == OpCode.PUSH else Role.SLAVE
    if protocol == Protocol.AXI4:
        return Role.SLAVE
    raise ValueError(f"Unknown protocol: {protocol}")
