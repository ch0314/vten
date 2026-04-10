"""Reporting: metadata bridge between CompiledResult and output.

Builds enriched command stats by joining IR Command metadata
with backend CmdStats, for human-readable and machine-parseable reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.spec.models import CommandStatus, OpCode

if TYPE_CHECKING:
    from vten.backend.base import CmdStats
    from vten.runtime.engine import CompiledResult
    from vten.runtime.ir import Command


# ── CommandMetadata ──


@dataclass
class CommandMetadata:
    """Static metadata for a single IR Command (no runtime stats)."""

    cmd_id: int
    op_name: str  # OpCode.name (e.g., "PUSH", "WRITE_REG")
    interface_name: str  # top-level interface name
    protocol: str  # Protocol.value (e.g., "axi4_stream")
    tensor_name: str | None  # exposed tensor name, None for reg/barrier ops
    size: int  # Command.size in bytes
    dep: list[int]
    commit_dep: list[int]
    sub_kernel: str | None  # sub-kernel name for CompositeKernel, None for "_self"
    origin_path: str | None  # e.g., "dma_ifm.src"
    port: str  # split port name, "" if not split
    probe: bool  # probe flag on command
    reg_offset: int  # register offset (for AXI4-Lite ops)
    reg_value: int  # register value (for writes)


# ── VerificationResult ──


@dataclass
class VerificationResult:
    """Structured result of one tensor verification."""

    tensor_name: str
    passed: bool
    max_diff: float = 0.0
    shape: tuple[int, ...] | None = None
    first_mismatch_index: list[int] | None = None
    expected_value: float | None = None
    actual_value: float | None = None


# ── ProbeResult ──


@dataclass
class ProbeResult:
    """Structured result of one probe point comparison."""

    probe_point: str  # e.g., "mac_atu.ifm_in"
    connection: str  # e.g., "fmapIO.ifm_out → mac_atu.ifm_in"
    passed: bool
    golden_shape: tuple[int, ...] | None = None
    golden_dtype: str | None = None
    max_diff: float = 0.0
    mismatch_count: int = 0
    first_mismatch_index: list[int] | None = None
    expected_value: float | None = None
    actual_value: float | None = None


# ── EnrichedCmdStats ──


@dataclass
class EnrichedCmdStats:
    """CmdStats + CommandMetadata merged, ready for JSON/terminal output."""

    # From CommandMetadata
    cmd_id: int
    op_name: str
    interface_name: str
    protocol: str
    tensor_name: str | None
    size: int
    dep: list[int]
    commit_dep: list[int]
    sub_kernel: str | None
    origin_path: str | None
    port: str
    probe: bool
    reg_offset: int
    reg_value: int

    # From CmdStats (runtime)
    status_code: int  # raw int from hardware
    status_name: str  # CommandStatus enum name
    issue_cycle: int
    commit_cycle: int
    active_cycles: int
    stall_cycles: int
    total_beats: int
    latency_cycles: int
    utilization: float
    bus_efficiency: float

    def to_dict(self) -> dict:
        """Serialize to JSON-ready dict."""
        d: dict = {
            "cmd_id": self.cmd_id,
            "op": self.op_name,
            "interface": self.interface_name,
            "protocol": self.protocol,
            "status": self.status_code,
            "status_name": self.status_name,
            "issue_cycle": self.issue_cycle,
            "commit_cycle": self.commit_cycle,
            "latency_cycles": self.latency_cycles,
            "active_cycles": self.active_cycles,
            "stall_cycles": self.stall_cycles,
            "total_beats": self.total_beats,
            "utilization": round(self.utilization, 4),
            "bus_efficiency": round(self.bus_efficiency, 4),
        }
        if self.tensor_name is not None:
            d["tensor"] = self.tensor_name
            d["size"] = self.size
        if self.sub_kernel and self.sub_kernel != "_self":
            d["sub_kernel"] = self.sub_kernel
            d["origin_path"] = self.origin_path
        if self.port:
            d["port"] = self.port
        if self.probe:
            d["probe"] = True
        if self.dep:
            d["dep"] = self.dep
        if self.commit_dep:
            d["commit_dep"] = self.commit_dep
        if self.op_name in ("WRITE_REG", "READ_REG", "POLL_REG"):
            d["reg_offset"] = self.reg_offset
            if self.op_name == "WRITE_REG":
                d["reg_value"] = self.reg_value
        return d


# ── Build functions ──


def _status_name(code: int) -> str:
    """Convert status int to CommandStatus enum name."""
    try:
        return CommandStatus(code).name
    except ValueError:
        return f"UNKNOWN({code})"


def build_command_metadata(compiled: CompiledResult) -> list[CommandMetadata]:
    """Extract metadata from CompiledResult for each Command.

    Maps interface_id → name, buffer_id → tensor_name, and
    for CompositeKernel, resolves sub_kernel from ExposedTensor.origin_path.
    """
    view = compiled.flattened_view

    # Reverse maps
    iface_id_to_name: dict[int, str] = {}
    if hasattr(compiled, "iface_id_to_name") and compiled.iface_id_to_name:
        iface_id_to_name = compiled.iface_id_to_name
    buffer_id_to_name: dict[int, str] = {
        bid: name for name, bid in compiled.buffer_ids.items()
    }

    # Ops that don't use a BFM interface (host↔memory, sync)
    _NO_INTERFACE_OPS = {OpCode.LOAD, OpCode.STORE, OpCode.BARRIER}

    result: list[CommandMetadata] = []
    for cmd in compiled.commands:
        # Interface name — LOAD/STORE/BARRIER don't use BFM interfaces
        if cmd.op in _NO_INTERFACE_OPS:
            iface_name = ""
        else:
            iface_name = iface_id_to_name.get(cmd.interface_id, "")

        # Tensor name from buffer_id
        tensor_name = buffer_id_to_name.get(cmd.buffer_id)
        # Register/barrier commands have buffer_id=0 which may collide
        if cmd.op in (
            OpCode.WRITE_REG, OpCode.READ_REG, OpCode.POLL_REG,
            OpCode.BARRIER,
        ):
            tensor_name = None

        # Sub-kernel and origin_path
        sub_kernel: str | None = None
        origin_path: str | None = None
        if tensor_name and tensor_name in view.exposed_tensors:
            exposed = view.exposed_tensors[tensor_name]
            origin_path = exposed.origin_path
            # origin_path format: "sub_kernel.tensor_name" or "_self.tensor_name"
            parts = origin_path.split(".", 1)
            sub_kernel = parts[0] if len(parts) > 1 else None

        # For register commands, derive sub_kernel from interface mapping
        if sub_kernel is None and iface_name:
            for m in view.interface_mappings:
                if m.top_interface == iface_name:
                    sub_kernel = m.sub_kernel
                    break

        result.append(CommandMetadata(
            cmd_id=cmd.cmd_id,
            op_name=cmd.op.name,
            interface_name=iface_name,
            protocol=cmd.protocol.value,
            tensor_name=tensor_name,
            size=cmd.size,
            dep=list(cmd.dep),
            commit_dep=list(cmd.commit_dep),
            sub_kernel=sub_kernel,
            origin_path=origin_path,
            port=cmd.port,
            probe=cmd.probe,
            reg_offset=cmd.reg_offset,
            reg_value=cmd.reg_value,
        ))

    return result


def merge_stats_with_metadata(
    stats: list[CmdStats],
    metadata: list[CommandMetadata],
) -> list[EnrichedCmdStats]:
    """Join CmdStats with CommandMetadata by cmd_id.

    Returns enriched stats list. If metadata is shorter than stats
    (e.g., pre-built SHM without CompiledResult), missing entries
    get default metadata.
    """
    meta_by_id: dict[int, CommandMetadata] = {m.cmd_id: m for m in metadata}

    result: list[EnrichedCmdStats] = []
    for s in stats:
        m = meta_by_id.get(s.cmd_id)
        result.append(EnrichedCmdStats(
            cmd_id=s.cmd_id,
            op_name=m.op_name if m else "",
            interface_name=m.interface_name if m else "",
            protocol=m.protocol if m else "",
            tensor_name=m.tensor_name if m else None,
            size=m.size if m else 0,
            dep=m.dep if m else [],
            commit_dep=m.commit_dep if m else [],
            sub_kernel=m.sub_kernel if m else None,
            origin_path=m.origin_path if m else None,
            port=m.port if m else "",
            probe=m.probe if m else False,
            reg_offset=m.reg_offset if m else 0,
            reg_value=m.reg_value if m else 0,
            status_code=s.status,
            status_name=_status_name(s.status),
            issue_cycle=s.issue_cycle,
            commit_cycle=s.commit_cycle,
            active_cycles=s.active_cycles,
            stall_cycles=s.stall_cycles,
            total_beats=s.total_beats,
            latency_cycles=s.latency_cycles,
            utilization=s.utilization,
            bus_efficiency=s.bus_efficiency,
        ))
    return result
