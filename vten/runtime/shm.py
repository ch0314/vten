"""Stage 7: SHM Packing.

SHM constants, BufferDescriptor, SHMBufferAllocator, and packing functions.

Spec reference: 00_data_models.md §11, 02_runtime_engine.md §14
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.spec.models import CommandStatus, Direction, OpCode, Protocol, Role

if TYPE_CHECKING:
    from vten.runtime.ir import Command

# ── SHM Constants ──

CONTROL_SIZE = 256
CMD_SLOT_SIZE = 64
STATS_SLOT_SIZE = 32
BUF_DESC_SIZE = 24
CACHE_LINE = 64

SHM_MAGIC = 0x5654454E  # "VTEN"
PROTOCOL_VERSION = 0x00000003

# Protocol encoding for SHM
PROTOCOL_ENCODING = {
    Protocol.AXI4S: 1,
    Protocol.AXI4: 2,
    Protocol.AXI4L: 3,
}

ROLE_ENCODING = {
    Role.MASTER: 0,
    Role.SLAVE: 1,
}

DIRECTION_ENCODING = {
    Direction.HOST_TO_DEV: 0,
    Direction.DEV_TO_HOST: 1,
    Direction.BIDIRECTIONAL: 2,
}

# Host status values (Control Region offset 0x08)
HOST_STATUS_IDLE = 0
HOST_STATUS_CMD_READY = 1
HOST_STATUS_ACK = 2
HOST_STATUS_SHUTDOWN = 3

# Backend status values (Control Region offset 0x0C)
BACKEND_STATUS_IDLE = 0
BACKEND_STATUS_RUNNING = 1
BACKEND_STATUS_DONE = 2
BACKEND_STATUS_ERROR = 3


# ── BufferDescriptor ──


@dataclass
class BufferDescriptor:
    buffer_id: int
    direction: int
    flags: int
    size: int
    data_offset: int


# ── SHMBufferAllocator ──


class SHMBufferAllocator:
    """Allocates data buffers with 64-byte alignment."""

    def __init__(self) -> None:
        self.next_offset = 0
        self.descriptors: list[BufferDescriptor] = []

    def allocate(
        self, buffer_id: int, size: int, direction: int, flags: int = 0
    ) -> int:
        aligned = (self.next_offset + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
        desc = BufferDescriptor(
            buffer_id=buffer_id,
            direction=direction,
            flags=flags,
            size=size,
            data_offset=aligned,
        )
        self.descriptors.append(desc)
        self.next_offset = aligned + size
        return aligned

    @property
    def total_data_size(self) -> int:
        return (self.next_offset + CACHE_LINE - 1) & ~(CACHE_LINE - 1)

    def get_descriptor(self, buffer_id: int) -> BufferDescriptor:
        for d in self.descriptors:
            if d.buffer_id == buffer_id:
                return d
        raise KeyError(f"Buffer {buffer_id} not found")


# ── Size calculation ──


def calculate_shm_size(
    num_commands: int, num_buffers: int, buffer_sizes: list[int]
) -> int:
    size = CONTROL_SIZE
    size += CMD_SLOT_SIZE * num_commands
    size += STATS_SLOT_SIZE * num_commands
    size += BUF_DESC_SIZE * num_buffers
    for buf_size in buffer_sizes:
        size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
        size += buf_size
    size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
    return size


# ── Packing functions ──


def pack_control_header(
    image: bytearray,
    num_commands: int,
    num_buffers: int,
    cmd_region_offset: int,
    stats_region_offset: int,
    buf_desc_offset: int,
    data_region_offset: int,
    total_shm_size: int,
) -> None:
    """Pack the 256-byte control header."""
    struct.pack_into("<I", image, 0x00, SHM_MAGIC)
    struct.pack_into("<I", image, 0x04, PROTOCOL_VERSION)
    struct.pack_into("<I", image, 0x08, 0)  # host_status = IDLE
    struct.pack_into("<I", image, 0x0C, 0)  # backend_status = IDLE
    struct.pack_into("<I", image, 0x10, num_commands)
    struct.pack_into("<I", image, 0x14, num_buffers)
    struct.pack_into("<Q", image, 0x18, cmd_region_offset)
    struct.pack_into("<Q", image, 0x20, stats_region_offset)
    struct.pack_into("<Q", image, 0x28, buf_desc_offset)
    struct.pack_into("<Q", image, 0x30, data_region_offset)
    struct.pack_into("<Q", image, 0x38, total_shm_size)
    # error_code, error_cmd_id, error_message, flags, timeout, freq, seq = 0
    struct.pack_into("<I", image, 0x88, 0x01)  # flags: STATS_ENABLED


def pack_command_slot(
    image: bytearray, offset: int, cmd: Command
) -> None:
    """Pack a single 64-byte command slot."""
    struct.pack_into("<H", image, offset + 0x00, cmd.op.value)
    struct.pack_into("<H", image, offset + 0x02, cmd.cmd_id)
    struct.pack_into("<H", image, offset + 0x04, cmd.interface_id)
    struct.pack_into("<B", image, offset + 0x06, PROTOCOL_ENCODING.get(cmd.protocol, 1))
    struct.pack_into("<B", image, offset + 0x07, ROLE_ENCODING.get(cmd.role, 0))
    struct.pack_into("<H", image, offset + 0x08, cmd.buffer_id)
    struct.pack_into("<B", image, offset + 0x0A, 1 if cmd.probe else 0)
    flags = 0x01 if cmd.sync else 0x00
    struct.pack_into("<B", image, offset + 0x0B, flags)
    struct.pack_into("<I", image, offset + 0x0C, cmd.size)
    struct.pack_into("<Q", image, offset + 0x10, cmd.phys_addr)
    struct.pack_into("<I", image, offset + 0x18, cmd.reg_offset)
    struct.pack_into("<I", image, offset + 0x1C, cmd.reg_value)
    struct.pack_into("<I", image, offset + 0x20, cmd.reg_mask)
    struct.pack_into("<I", image, offset + 0x24, cmd.reg_expected)
    struct.pack_into("<H", image, offset + 0x28, cmd.golden_buf)

    # Dependencies
    num_deps = len(cmd.dep)
    num_commit_deps = len(cmd.commit_dep)
    struct.pack_into("<B", image, offset + 0x2A, num_deps)
    struct.pack_into("<B", image, offset + 0x2B, num_commit_deps)

    # dep_ids: 4 x 2B, unused = 0xFFFF
    for i in range(4):
        val = cmd.dep[i] if i < num_deps else 0xFFFF
        struct.pack_into("<H", image, offset + 0x2C + i * 2, val)

    # commit_dep_ids: 4 x 2B, unused = 0xFFFF
    for i in range(4):
        val = cmd.commit_dep[i] if i < num_commit_deps else 0xFFFF
        struct.pack_into("<H", image, offset + 0x34 + i * 2, val)

    # reserved
    struct.pack_into("<I", image, offset + 0x3C, 0)


def pack_stats_entry(
    image: bytearray, offset: int, status: int = 0
) -> None:
    """Pack a 32-byte stats entry."""
    struct.pack_into("<B", image, offset + 0x00, status)
    # Rest is zeros (already initialized)


def pack_buffer_descriptor(
    image: bytearray, offset: int, desc: BufferDescriptor
) -> None:
    """Pack a 24-byte buffer descriptor."""
    struct.pack_into("<H", image, offset + 0x00, desc.buffer_id)
    struct.pack_into("<B", image, offset + 0x02, desc.direction)
    struct.pack_into("<B", image, offset + 0x03, desc.flags)
    struct.pack_into("<I", image, offset + 0x04, desc.size)
    struct.pack_into("<Q", image, offset + 0x08, desc.data_offset)
    struct.pack_into("<Q", image, offset + 0x10, 0)  # reserved
