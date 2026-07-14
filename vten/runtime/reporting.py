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


# ── Performance summary — roofline / utilization aggregation ──


@dataclass
class InterfacePerf:
    """Aggregated performance for a single interface."""

    interface: str
    protocol: str
    commands: int
    total_beats: int
    active_cycles: int
    stall_cycles: int
    latency_cycles: int
    active_window: int  # last_active - first_active + 1 across the interface
    bytes_moved: int  # from Command.size, 0 when unknown
    utilization: float  # active_cycles / active_window
    bus_efficiency: float  # active_cycles / latency_cycles
    beats_per_cycle: float  # total_beats / active_window
    bytes_per_cycle: float  # bytes_moved / active_window (0 when bytes unknown)
    bytes_per_beat: float  # bytes_moved / total_beats (0 when unknown)

    def to_dict(self) -> dict:
        return {
            "interface": self.interface,
            "protocol": self.protocol,
            "commands": self.commands,
            "total_beats": self.total_beats,
            "active_cycles": self.active_cycles,
            "stall_cycles": self.stall_cycles,
            "latency_cycles": self.latency_cycles,
            "active_window": self.active_window,
            "bytes_moved": self.bytes_moved,
            "utilization": round(self.utilization, 4),
            "bus_efficiency": round(self.bus_efficiency, 4),
            "beats_per_cycle": round(self.beats_per_cycle, 4),
            "bytes_per_cycle": round(self.bytes_per_cycle, 4),
            "bytes_per_beat": round(self.bytes_per_beat, 4),
        }


@dataclass
class PerfSummary:
    """Per-run performance summary: per-interface + overall roofline view."""

    interfaces: list[InterfacePerf]
    total_beats: int
    active_cycles: int
    stall_cycles: int
    active_window: int  # span from earliest first_active to latest last_active
    bytes_moved: int
    utilization: float  # aggregate active / window
    bus_efficiency: float  # aggregate active / total latency
    beats_per_cycle: float
    bytes_per_cycle: float
    bottleneck_interface: str | None  # highest stall / lowest efficiency
    bottleneck_reason: str | None  # human-readable why
    clock_freq_hz: int | None = None  # None → report per-cycle, not per-second
    achieved_bandwidth_bps: float | None = None  # bytes/s when clock known

    def to_dict(self) -> dict:
        d: dict = {
            "interfaces": [i.to_dict() for i in self.interfaces],
            "overall": {
                "total_beats": self.total_beats,
                "active_cycles": self.active_cycles,
                "stall_cycles": self.stall_cycles,
                "active_window": self.active_window,
                "bytes_moved": self.bytes_moved,
                "utilization": round(self.utilization, 4),
                "bus_efficiency": round(self.bus_efficiency, 4),
                "beats_per_cycle": round(self.beats_per_cycle, 4),
                "bytes_per_cycle": round(self.bytes_per_cycle, 4),
                "bottleneck_interface": self.bottleneck_interface,
                "bottleneck_reason": self.bottleneck_reason,
            },
        }
        if self.clock_freq_hz is not None:
            d["overall"]["clock_freq_hz"] = self.clock_freq_hz
        if self.achieved_bandwidth_bps is not None:
            d["overall"]["achieved_bandwidth_bps"] = round(
                self.achieved_bandwidth_bps, 2
            )
        return d


# Ops that move data over a bus (contribute to bandwidth/utilization).
_DATA_OPS = frozenset({"PUSH", "PULL", "LOAD", "STORE"})


def _cmd_field(cmd: object, name: str, default: int = 0) -> int:
    """Read a numeric field from either a dict (report schema) or an object."""
    if isinstance(cmd, dict):
        val = cmd.get(name, default)
    else:
        val = getattr(cmd, name, default)
    return val if val is not None else default


def _cmd_str(cmd: object, *names: str) -> str:
    """Read the first non-empty string field from a dict/object."""
    for name in names:
        if isinstance(cmd, dict):
            val = cmd.get(name)
        else:
            val = getattr(cmd, name, None)
        if val:
            return str(val)
    return ""


def build_perf_summary(
    commands: list,
    clock_freq_hz: int | None = None,
) -> PerfSummary | None:
    """Aggregate per-command stats into a per-interface performance summary.

    Accepts either enriched command dicts (the stats.json ``commands`` schema
    produced by :meth:`EnrichedCmdStats.to_dict`) or ``EnrichedCmdStats``
    objects. Only data-moving ops (PUSH/PULL/LOAD/STORE) with cycle timing
    contribute; register/barrier/sync ops are ignored.

    Per interface it reports beats, active/stall cycles, utilization,
    bus efficiency, bytes moved and achieved bandwidth (beats/cycle and, when
    ``size`` bytes are present, bytes/cycle). The overall section adds the
    bottleneck interface — the one with the highest stall cycles, breaking
    ties by lowest bus efficiency.

    Reuses the same math as ``CmdStats.utilization`` / ``bus_efficiency``:
    utilization = active_cycles / active_window,
    bus_efficiency = active_cycles / latency_cycles.

    Returns ``None`` when no command carries per-command cycle stats (e.g.
    the cpu backend, which emits no CmdStats) so callers can degrade
    gracefully rather than rendering an empty table.
    """
    # Accumulators keyed by interface name.
    agg: dict[str, dict] = {}
    any_stats = False

    for cmd in commands:
        op = _cmd_str(cmd, "op", "op_name")
        if op and op not in _DATA_OPS:
            continue

        beats = _cmd_field(cmd, "total_beats")
        active = _cmd_field(cmd, "active_cycles")
        latency = _cmd_field(cmd, "latency_cycles")
        stall = _cmd_field(cmd, "stall_cycles")
        first_active = _cmd_field(cmd, "first_active_cycle", -1)
        last_active = _cmd_field(cmd, "last_active_cycle", -1)

        # A command counts as "having stats" if it moved beats or spent
        # active/latency cycles. Pure metadata rows (all zero) don't.
        if not (beats or active or latency or stall):
            continue
        any_stats = True

        iface = _cmd_str(cmd, "interface", "interface_name") or "(none)"
        proto = _cmd_str(cmd, "protocol")

        a = agg.setdefault(iface, {
            "protocol": proto,
            "commands": 0,
            "total_beats": 0,
            "active_cycles": 0,
            "stall_cycles": 0,
            "latency_cycles": 0,
            "bytes_moved": 0,
            "first_active": None,
            "last_active": None,
        })
        if not a["protocol"] and proto:
            a["protocol"] = proto
        a["commands"] += 1
        a["total_beats"] += beats
        a["active_cycles"] += active
        a["stall_cycles"] += stall
        a["latency_cycles"] += latency
        a["bytes_moved"] += _cmd_field(cmd, "size")

        if first_active >= 0:
            a["first_active"] = (
                first_active if a["first_active"] is None
                else min(a["first_active"], first_active)
            )
        if last_active >= 0:
            a["last_active"] = (
                last_active if a["last_active"] is None
                else max(a["last_active"], last_active)
            )

    if not any_stats:
        return None

    interfaces: list[InterfacePerf] = []
    for name in sorted(agg):
        a = agg[name]
        # Active window: prefer measured first/last active span; fall back to
        # summed active cycles when per-command first/last are unavailable.
        if a["first_active"] is not None and a["last_active"] is not None:
            window = a["last_active"] - a["first_active"] + 1
        else:
            window = a["active_cycles"]
        window = max(window, 0)

        util = a["active_cycles"] / window if window else 0.0
        eff = (
            a["active_cycles"] / a["latency_cycles"]
            if a["latency_cycles"] else 0.0
        )
        bpc = a["total_beats"] / window if window else 0.0
        bytes_pc = a["bytes_moved"] / window if window else 0.0
        bytes_pb = (
            a["bytes_moved"] / a["total_beats"] if a["total_beats"] else 0.0
        )

        interfaces.append(InterfacePerf(
            interface=name,
            protocol=a["protocol"],
            commands=a["commands"],
            total_beats=a["total_beats"],
            active_cycles=a["active_cycles"],
            stall_cycles=a["stall_cycles"],
            latency_cycles=a["latency_cycles"],
            active_window=window,
            bytes_moved=a["bytes_moved"],
            utilization=util,
            bus_efficiency=eff,
            beats_per_cycle=bpc,
            bytes_per_cycle=bytes_pc,
            bytes_per_beat=bytes_pb,
        ))

    # ── Overall aggregate ──
    total_beats = sum(i.total_beats for i in interfaces)
    total_active = sum(i.active_cycles for i in interfaces)
    total_stall = sum(i.stall_cycles for i in interfaces)
    total_latency = sum(i.latency_cycles for i in interfaces)
    total_bytes = sum(i.bytes_moved for i in interfaces)

    first_actives = [
        agg[n]["first_active"] for n in agg
        if agg[n]["first_active"] is not None
    ]
    last_actives = [
        agg[n]["last_active"] for n in agg
        if agg[n]["last_active"] is not None
    ]
    if first_actives and last_actives:
        overall_window = max(last_actives) - min(first_actives) + 1
    else:
        overall_window = total_active
    overall_window = max(overall_window, 0)

    overall_util = total_active / overall_window if overall_window else 0.0
    overall_eff = total_active / total_latency if total_latency else 0.0
    overall_bpc = total_beats / overall_window if overall_window else 0.0
    overall_bytes_pc = total_bytes / overall_window if overall_window else 0.0

    # ── Bottleneck: highest stall cycles; tie-break on lowest efficiency ──
    bottleneck = None
    reason = None
    if interfaces:
        bottleneck_if = max(
            interfaces,
            key=lambda i: (i.stall_cycles, -i.bus_efficiency),
        )
        bottleneck = bottleneck_if.interface
        if bottleneck_if.stall_cycles > 0:
            reason = (
                f"{bottleneck_if.stall_cycles} stall cycles, "
                f"{bottleneck_if.bus_efficiency * 100:.1f}% bus efficiency"
            )
        else:
            # No stalls anywhere → flag the least efficient interface instead.
            least_eff = min(interfaces, key=lambda i: i.bus_efficiency)
            bottleneck = least_eff.interface
            reason = (
                f"lowest bus efficiency "
                f"({least_eff.bus_efficiency * 100:.1f}%)"
            )

    achieved_bw = None
    if clock_freq_hz and overall_window and total_bytes:
        # bytes/cycle × cycles/s = bytes/s
        achieved_bw = overall_bytes_pc * clock_freq_hz

    return PerfSummary(
        interfaces=interfaces,
        total_beats=total_beats,
        active_cycles=total_active,
        stall_cycles=total_stall,
        active_window=overall_window,
        bytes_moved=total_bytes,
        utilization=overall_util,
        bus_efficiency=overall_eff,
        beats_per_cycle=overall_bpc,
        bytes_per_cycle=overall_bytes_pc,
        bottleneck_interface=bottleneck,
        bottleneck_reason=reason,
        clock_freq_hz=clock_freq_hz,
        achieved_bandwidth_bps=achieved_bw,
    )


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
