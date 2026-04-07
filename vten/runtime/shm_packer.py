"""SHM image packing — Stage 7 of the compile pipeline.

Extracted from RuntimeEngine to keep engine.py focused on IR generation (Stages 0–6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vten.spec.models import OpCode
from vten.runtime.shm import (
    BUF_DESC_SIZE,
    CACHE_LINE,
    CMD_SLOT_SIZE,
    CONTROL_SIZE,
    DIRECTION_ENCODING,
    STATS_SLOT_SIZE,
    SHMBufferAllocator,
    calculate_shm_size,
    pack_buffer_descriptor,
    pack_command_slot,
    pack_control_header,
    pack_stats_entry,
)

if TYPE_CHECKING:
    from vten.dsl.operations import Operation
    from vten.runtime.flattener import ExposedTensor, FlattenedKernelView
    from vten.runtime.ir import Command


# ── Layout metadata ──


@dataclass
class SHMLayout:
    """SHM region offsets and counts for in-place batch updates."""

    cmd_offset: int
    stats_offset: int
    bufdesc_offset: int
    data_region_offset: int
    num_commands: int
    num_buffers: int
    total_size: int


# ── Internal helpers ──


def _pack_shm_common(
    shm_alloc: SHMBufferAllocator,
    commands: list[Command],
    *,
    flags: int = 0,
) -> tuple[bytearray, SHMLayout, int]:
    """Pack SHM header, command slots, stats, and buffer descriptors.

    Shared by pack_shm (single-config) and pack_shm_multi (multi-config).
    Callers are responsible for allocating data buffers into *shm_alloc*
    before calling, and for copying tensor data into the returned image
    after this function returns.

    Returns:
        (image, layout, data_region_offset)
    """
    from vten.runtime.shm import FLAG_STATS_ENABLED
    from vten.spec.models import CommandStatus

    num_commands = len(commands)
    num_buffers = len(shm_alloc.descriptors)

    total = calculate_shm_size(
        num_commands=num_commands,
        num_buffers=num_buffers,
        buffer_sizes=[d.size for d in shm_alloc.descriptors],
    )

    image = bytearray(total)

    cmd_offset = CONTROL_SIZE
    stats_offset = cmd_offset + CMD_SLOT_SIZE * num_commands
    bufdesc_offset = stats_offset + STATS_SLOT_SIZE * num_commands
    data_region_raw = bufdesc_offset + BUF_DESC_SIZE * num_buffers
    data_region_offset = (data_region_raw + CACHE_LINE - 1) & ~(CACHE_LINE - 1)

    layout = SHMLayout(
        cmd_offset=cmd_offset,
        stats_offset=stats_offset,
        bufdesc_offset=bufdesc_offset,
        data_region_offset=data_region_offset,
        num_commands=num_commands,
        num_buffers=num_buffers,
        total_size=total,
    )

    shm_flags = flags | FLAG_STATS_ENABLED
    pack_control_header(
        image,
        num_commands=num_commands,
        num_buffers=num_buffers,
        cmd_region_offset=cmd_offset,
        stats_region_offset=stats_offset,
        buf_desc_offset=bufdesc_offset,
        data_region_offset=data_region_offset,
        total_shm_size=total,
        flags=shm_flags,
    )

    for i, cmd in enumerate(commands):
        pack_command_slot(image, cmd_offset + i * CMD_SLOT_SIZE, cmd)

    for cmd in commands:
        if cmd.op == OpCode.LOAD:
            pack_stats_entry(
                image,
                stats_offset + cmd.cmd_id * STATS_SLOT_SIZE,
                status=CommandStatus.COMMITTED.value,
            )

    for i, desc in enumerate(shm_alloc.descriptors):
        pack_buffer_descriptor(
            image, bufdesc_offset + i * BUF_DESC_SIZE, desc
        )

    return image, layout, data_region_offset


def _parse_split_spec(raw):
    """Parse a raw dict or SplitSpec into a SplitSpec dataclass."""
    from vten.spec.models import InterleaveSpec, PortDef, SplitSpec

    if isinstance(raw, SplitSpec):
        return raw
    ports = [
        PortDef(name=p["name"], base_addr=p.get("base_addr", 0))
        for p in raw.get("ports", [])
    ]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)


def _block_split_data(
    serialized: bytes | None,
    flat_names: list[str],
    serialized_size: int,
) -> dict[str, bytes]:
    """Block-split serialized data (or allocate empty) across port names."""
    n = len(flat_names)
    if serialized is not None:
        data = serialized
        chunk_size = len(data) // n
        remainder = len(data) % n
        result = {}
        offset = 0
        for i, fname in enumerate(flat_names):
            sz = chunk_size + (1 if i < remainder else 0)
            result[fname] = data[offset : offset + sz]
            offset += sz
        return result
    else:
        per_elem_size = serialized_size // n
        return {fname: bytes(per_elem_size) for fname in flat_names}


# ── Public packing API ──


def pack_shm(
    view: FlattenedKernelView,
    commands: list[Command],
    buffer_ids: dict[str, int],
    ops: list[Operation],
    flags: int = 0,
) -> tuple[bytes, SHMLayout]:
    """Pack SHM image for a single-config compile.

    Returns:
        (shm_image, layout) — packed bytes and region-offset metadata.
    """
    shm_alloc = SHMBufferAllocator()
    allocated_buffer_ids: set[int] = set()

    # Collect chunk info from ops to detect chunked tensors
    chunk_tensors: dict[str, int | list[int]] = {}
    for op in ops:
        if op.chunk_total is not None and op.tensor is not None:
            chunk_tensors[op.tensor.name] = op.chunks_spec

    for name, exposed in view.exposed_tensors.items():
        direction = DIRECTION_ENCODING.get(exposed.direction, 0)

        if name in chunk_tensors:
            chunks_spec = chunk_tensors[name]
            n_chunks = (
                len(chunks_spec) if isinstance(chunks_spec, list)
                else chunks_spec
            )
            for ci in range(n_chunks):
                if isinstance(chunks_spec, list):
                    total_elems = sum(chunks_spec)
                    chunk_size = (
                        exposed._serialized_size * chunks_spec[ci]
                        // total_elems
                    )
                else:
                    chunk_size = exposed._serialized_size // n_chunks

                if exposed._port_buffers:
                    flat_names = list(exposed._port_buffers.keys())
                    n_elems = len(flat_names)
                    per_elem_size = chunk_size // n_elems
                    for fname in flat_names:
                        bid = buffer_ids[f"{name}:chunk_{ci}:{fname}"]
                        if bid not in allocated_buffer_ids:
                            allocated_buffer_ids.add(bid)
                            shm_alloc.allocate(bid, per_elem_size, direction)
                else:
                    bid = buffer_ids[f"{name}:chunk_{ci}"]
                    if bid not in allocated_buffer_ids:
                        allocated_buffer_ids.add(bid)
                        shm_alloc.allocate(bid, chunk_size, direction)
        elif exposed._port_buffers:
            for flat_name, chunk in exposed._port_buffers.items():
                bid = buffer_ids[f"{name}:{flat_name}"]
                if bid not in allocated_buffer_ids:
                    allocated_buffer_ids.add(bid)
                    shm_alloc.allocate(bid, len(chunk), direction)
        else:
            bid = buffer_ids[name]
            if bid not in allocated_buffer_ids:
                allocated_buffer_ids.add(bid)
                shm_alloc.allocate(bid, exposed._serialized_size, direction)

    # Probe golden buffers
    next_buffer_id = max(buffer_ids.values(), default=-1) + 1
    probe_port_golden: dict[int, tuple[int, bytes]] = {}
    for probe in view.probe_points:
        if probe.serialized_golden is None:
            continue
        exposed = (
            view.exposed_tensors.get(probe.tensor_name)
            if probe.tensor_name else None
        )
        if exposed and exposed._port_buffers:
            golden_bytes = probe.serialized_golden
            offset = 0
            for port_name, port_data in exposed._port_buffers.items():
                port_size = len(port_data)
                golden_chunk = golden_bytes[offset:offset + port_size]
                shm_alloc.allocate(next_buffer_id, port_size, 0, flags=0x01)
                port_bid = buffer_ids.get(f"{probe.tensor_name}:{port_name}")
                if port_bid is not None:
                    probe_port_golden[port_bid] = (next_buffer_id, golden_chunk)
                next_buffer_id += 1
                offset += port_size
        else:
            shm_alloc.allocate(
                next_buffer_id, len(probe.serialized_golden), 0, flags=0x01
            )
            probe.golden_buffer_id = next_buffer_id
            next_buffer_id += 1

    # Assign cmd.golden_buf for probe PULL commands
    probe_tensor_bids = {
        p.tensor_name: p.golden_buffer_id
        for p in view.probe_points
        if p.tensor_name and p.golden_buffer_id is not None
    }
    for cmd in commands:
        if cmd.probe and cmd.op == OpCode.PULL:
            if cmd.buffer_id in probe_port_golden:
                cmd.golden_buf = probe_port_golden[cmd.buffer_id][0]
            elif probe_tensor_bids:
                for tname, bid in buffer_ids.items():
                    base = tname.split(":")[0] if ":" in tname else tname
                    if bid == cmd.buffer_id and base in probe_tensor_bids:
                        cmd.golden_buf = probe_tensor_bids[base]
                        break

    image, layout, data_region_offset = _pack_shm_common(
        shm_alloc, commands, flags=flags
    )

    # Copy input tensor data
    for name, exposed in view.exposed_tensors.items():
        if exposed._port_buffers:
            for flat_name, chunk in exposed._port_buffers.items():
                if not chunk or all(b == 0 for b in chunk):
                    continue
                bid = buffer_ids[f"{name}:{flat_name}"]
                try:
                    desc = shm_alloc.get_descriptor(bid)
                except KeyError:
                    continue
                start = data_region_offset + desc.data_offset
                image[start : start + len(chunk)] = chunk
        elif exposed._serialized is not None:
            bid = buffer_ids[name]
            try:
                desc = shm_alloc.get_descriptor(bid)
            except KeyError:
                continue
            start = data_region_offset + desc.data_offset
            image[start : start + len(exposed._serialized)] = exposed._serialized

    # Copy probe golden data
    for probe in view.probe_points:
        if probe.serialized_golden is not None and probe.golden_buffer_id is not None:
            desc = shm_alloc.get_descriptor(probe.golden_buffer_id)
            start = data_region_offset + desc.data_offset
            image[start : start + len(probe.serialized_golden)] = probe.serialized_golden
    for _port_bid, (golden_bid, golden_chunk) in probe_port_golden.items():
        desc = shm_alloc.get_descriptor(golden_bid)
        start = data_region_offset + desc.data_offset
        image[start : start + len(golden_chunk)] = golden_chunk

    return bytes(image), layout


def pack_shm_multi(
    views: list[FlattenedKernelView],
    view_buffer_ids: list[dict[str, int]],
    commands: list[Command],
) -> bytes:
    """Pack SHM image for a multi-config batch."""
    shm_alloc = SHMBufferAllocator()
    allocated_buffer_ids: set[int] = set()

    for view, buffer_ids in zip(views, view_buffer_ids):
        for name, exposed in view.exposed_tensors.items():
            direction = DIRECTION_ENCODING.get(exposed.direction, 0)

            if exposed._port_buffers:
                for flat_name, chunk in exposed._port_buffers.items():
                    bid = buffer_ids[f"{name}:{flat_name}"]
                    if bid not in allocated_buffer_ids:
                        allocated_buffer_ids.add(bid)
                        shm_alloc.allocate(bid, len(chunk), direction)
            else:
                bid = buffer_ids[name]
                if bid not in allocated_buffer_ids:
                    allocated_buffer_ids.add(bid)
                    shm_alloc.allocate(bid, exposed._serialized_size, direction)

    image, _layout, data_region_offset = _pack_shm_common(shm_alloc, commands)

    for view, buffer_ids in zip(views, view_buffer_ids):
        for name, exposed in view.exposed_tensors.items():
            if exposed._port_buffers:
                for flat_name, chunk in exposed._port_buffers.items():
                    if not chunk or all(b == 0 for b in chunk):
                        continue
                    bid = buffer_ids[f"{name}:{flat_name}"]
                    try:
                        desc = shm_alloc.get_descriptor(bid)
                    except KeyError:
                        continue
                    start = data_region_offset + desc.data_offset
                    image[start : start + len(chunk)] = chunk
            elif exposed._serialized is not None:
                bid = buffer_ids[name]
                try:
                    desc = shm_alloc.get_descriptor(bid)
                except KeyError:
                    continue
                start = data_region_offset + desc.data_offset
                image[start : start + len(exposed._serialized)] = exposed._serialized

    return bytes(image)