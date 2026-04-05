"""CommandInterpreter — IR Command to XRT API translator.

The HW counterpart to SHM Packer (Stage 7). While SHM Packer serializes
commands into a binary image for the hardware scheduler, CommandInterpreter
executes them sequentially on the host via XRT API calls.

Uses xrt.ip (raw register access) for maximum compatibility with RTL designs.

Spec reference: 08_backend_abstraction.md §4.3
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from vten.errors import BackendError, PollTimeoutError
from vten.log import (
    PHASE_CONFIGURE,
    PHASE_OTHER,
    PHASE_POLL,
    PHASE_RECV,
    PHASE_SEND,
    PHASE_TRIGGER,
    format_elapsed,
    format_size,
)
from vten.runtime.binder import parse_bit_range
from vten.spec.models import OpCode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.runtime.ir import Command


def _classify_phases(commands: list[Command]) -> list[str]:
    """Assign a phase label to each command.

    Phase order: configure → send → trigger → poll → recv.
    WRITE_REGs before any LOAD/PUSH are 'configure'.
    WRITE_REGs after LOAD/PUSH are 'trigger' (vsync).
    """
    phases: list[str] = []
    seen_data_op = False
    for cmd in commands:
        if cmd.op == OpCode.WRITE_REG:
            phases.append(PHASE_TRIGGER if seen_data_op else PHASE_CONFIGURE)
        elif cmd.op in (OpCode.LOAD, OpCode.PUSH):
            seen_data_op = True
            phases.append(PHASE_SEND)
        elif cmd.op in (OpCode.PULL, OpCode.STORE):
            phases.append(PHASE_RECV)
        elif cmd.op == OpCode.POLL_REG:
            phases.append(PHASE_POLL)
        else:
            phases.append(PHASE_OTHER)
    return phases


def _phase_summary(phase: str, commands: list[Command], phases: list[str]) -> str:
    """Build a one-line summary for a phase header."""
    cmds = [c for c, p in zip(commands, phases) if p == phase]
    n = len(cmds)
    if phase == PHASE_CONFIGURE:
        return f"{n} WRITE_REG"
    elif phase == PHASE_SEND:
        n_load = sum(1 for c in cmds if c.op == OpCode.LOAD)
        n_push = sum(1 for c in cmds if c.op == OpCode.PUSH)
        total_bytes = sum(c.size for c in cmds if c.op == OpCode.LOAD and c.size)
        return f"{n_load} LOAD + {n_push} PUSH, {format_size(total_bytes)}"
    elif phase == PHASE_TRIGGER:
        return f"{n} WRITE_REG vsync"
    elif phase == PHASE_POLL:
        return f"{n} POLL_REG"
    elif phase == PHASE_RECV:
        n_pull = sum(1 for c in cmds if c.op == OpCode.PULL)
        total_bytes = sum(c.size for c in cmds if c.op == OpCode.PULL and c.size)
        return f"{n_pull} tensor, {format_size(total_bytes)}"
    return f"{n} commands"


# ── Heartbeat / stall thresholds ──

_QUIET_PERIOD = 4.0        # seconds before heartbeat kicks in per phase
_POLL_REPORT_INTERVAL = 2.0  # seconds between POLL progress reports
_STALL_THRESHOLD = 10.0    # seconds of no change → WARNING


class CommandInterpreter:
    """Interprets IR Commands and executes them via XRT API.

    OpCode mapping:
      LOAD      → BO.write(data) — host writes tensor data into device buffer
      PUSH      → BO.sync(TO_DEVICE) — transfer to FPGA
      PULL      → BO.sync(FROM_DEVICE) — transfer from FPGA
      STORE     → BO.read(size) — host reads device buffer
      WRITE_REG → ip.write_register(offset, value)
      READ_REG  → ip.read_register(offset)
      POLL_REG  → polling loop on read_register
      BARRIER   → host-side fence (all pending ops complete)
      COMPARE   → host-side buffer comparison
    """

    def __init__(
        self,
        device: Any,
        kernel: Any,
        xrt_module: Any,
        arg_map: dict[int, int] | None = None,
        poll_timeout_ms: int = 10000,
        ip_map: dict[int, Any] | None = None,
        mem_bank_map: dict[int, int] | None = None,
        addr_bindings: (
            dict[tuple[int, int], tuple[int, str | None]] | None
        ) = None,
    ) -> None:
        self._device = device
        self._kernel = kernel
        self._xrt = xrt_module
        self._arg_map = arg_map or {}
        self._poll_timeout_ms = poll_timeout_ms
        self._ip_map = ip_map or {}
        self._mem_bank_map = mem_bank_map or {}
        self._addr_bindings = addr_bindings or {}
        self._buffers: dict[int, Any] = {}  # buffer_id → XRT BO
        self._output_buffers: dict[int, bytes] = {}
        self._completed: set[int] = set()  # completed cmd_ids
        self._prebound: set[int] = set()  # buffer_ids with pre-synced BOs
        self._deferred_stores: list[Command] = []  # STORE cmds deferred past POLL

    def update_maps(
        self,
        ip_map: dict[int, Any],
        mem_bank_map: dict[int, int],
        addr_bindings: dict,
    ) -> None:
        """Update maps for a new execution while preserving BO pool."""
        self._ip_map = ip_map
        self._mem_bank_map = mem_bank_map
        self._addr_bindings = addr_bindings
        self._output_buffers.clear()
        self._completed.clear()
        self._prebound.clear()
        self._deferred_stores.clear()

    def _get_ip(self, interface_id: int) -> Any:
        """Get xrt.ip for a given interface_id, fallback to default."""
        return self._ip_map.get(interface_id, self._kernel)

    @property
    def output_buffers(self) -> dict[int, bytes]:
        """Output buffers populated by STORE commands."""
        return dict(self._output_buffers)

    # ── Main execution loop ──

    def execute(
        self,
        commands: list[Command],
        tensor_data: dict[int, bytes],
    ) -> None:
        """Execute IR commands sequentially with phase-based logging.

        INFO level: phase headers + summaries (clean, concise).
        DEBUG level: individual command execution trace.
        """
        self._output_buffers.clear()
        self._completed.clear()

        # Pre-allocate BOs for auto_bind address resolution
        self._preallocate_bound_bos(commands, tensor_data)

        # Collect WRITE_REG coverage for addr_bindings
        covered_regs: set[tuple[int, int]] = set()
        for cmd in commands:
            if cmd.op == OpCode.WRITE_REG:
                covered_regs.add((cmd.interface_id, cmd.reg_offset))

        # Classify commands into phases
        phases = _classify_phases(commands)

        # Pre-compute per-phase byte totals for progress tracking
        phase_total_bytes: dict[str, int] = {}
        for cmd, ph in zip(commands, phases):
            if cmd.op == OpCode.LOAD and cmd.size:
                phase_total_bytes[ph] = phase_total_bytes.get(ph, 0) + cmd.size
            elif cmd.op == OpCode.PULL and cmd.size:
                phase_total_bytes[ph] = phase_total_bytes.get(ph, 0) + cmd.size

        emit_timeline = logger.isEnabledFor(logging.DEBUG)

        t0 = time.monotonic()
        addr_written = False
        current_phase: str | None = None
        phase_start = t0
        phase_bytes_done = 0
        phase_cmds_done = 0
        for i, cmd in enumerate(commands):
            # Write uncovered auto_bind address registers at configure→send transition
            if not addr_written and cmd.op != OpCode.WRITE_REG:
                self._write_addr_bindings(covered_regs)
                addr_written = True

            # Phase transition
            cmd_phase = phases[i]
            if cmd_phase != current_phase:
                # Log previous phase completion
                if current_phase is not None:
                    phase_elapsed = time.monotonic() - phase_start
                    self._log_phase_done(
                        current_phase, phase_elapsed,
                        phase_bytes_done, phase_total_bytes.get(current_phase, 0),
                        phase_cmds_done, emit_timeline, t0,
                    )
                # Start new phase
                current_phase = cmd_phase
                phase_start = time.monotonic()
                phase_bytes_done = 0
                phase_cmds_done = 0
                summary = _phase_summary(cmd_phase, commands, phases)
                logger.info("── %s (%s) ──", cmd_phase, summary)

            # Individual command trace (DEBUG only)
            op_name = cmd.op.name if hasattr(cmd.op, "name") else str(cmd.op)
            logger.debug(
                "  [%d/%d] %s iface=%s buf=%s",
                i + 1, len(commands), op_name,
                getattr(cmd, "interface_id", "-"),
                getattr(cmd, "buffer_id", "-"),
            )

            self._wait_deps(cmd)
            self._dispatch(cmd, tensor_data)
            self._completed.add(cmd.cmd_id)
            phase_cmds_done += 1

            # Track bytes for progress
            if cmd.op == OpCode.LOAD and cmd.size:
                phase_bytes_done += cmd.size
            elif cmd.op == OpCode.PULL and cmd.size:
                phase_bytes_done += cmd.size

            # JSON timeline entry
            if emit_timeline:
                self._emit_timeline(cmd, cmd_phase, time.monotonic() - t0)

        # Log final phase completion
        if current_phase is not None:
            phase_elapsed = time.monotonic() - phase_start
            self._log_phase_done(
                current_phase, phase_elapsed,
                phase_bytes_done, phase_total_bytes.get(current_phase, 0),
                phase_cmds_done, emit_timeline, t0,
            )

        # Flush deferred stores — always after all commands complete
        if self._deferred_stores:
            logger.info("── recv (flush) ──")
            self._flush_deferred_stores()

        total_elapsed = time.monotonic() - t0
        logger.info("execution complete: %s, %d commands",
                     format_elapsed(total_elapsed), len(commands))

    # ── addr_bindings ──

    def _write_addr_bindings(
        self, covered_regs: set[tuple[int, int]] | None = None,
    ) -> None:
        """Write uncovered auto_bind address registers to their IPs."""
        covered = covered_regs or set()
        written = 0
        skipped = 0
        for (iface_id, reg_offset), binding in self._addr_bindings.items():
            if (iface_id, reg_offset) in covered:
                skipped += 1
                continue
            if len(binding) == 3:
                buffer_id, bits_spec, byte_offset = binding
            else:
                buffer_id, bits_spec = binding
                byte_offset = 0
            bo = self._buffers.get(buffer_id)
            if bo is None or not hasattr(bo, "address"):
                continue
            addr = bo.address() + byte_offset
            if bits_spec:
                hi, lo = parse_bit_range(bits_spec)
                addr = (addr >> lo) & ((1 << (hi - lo + 1)) - 1)
            ip = self._get_ip(iface_id)
            ip.write_register(reg_offset, addr)
            written += 1
        if self._addr_bindings:
            logger.debug(
                "  addr_bindings: wrote %d, skipped %d (covered by WRITE_REG)",
                written, skipped,
            )

    # ── Phase progress & timeline helpers ──

    def _log_phase_done(
        self,
        phase: str,
        elapsed: float,
        bytes_done: int,
        bytes_total: int,
        cmds_done: int,
        emit_timeline: bool,
        t0: float,
    ) -> None:
        """Log phase completion with optional byte progress."""
        if elapsed >= _QUIET_PERIOD:
            if bytes_total > 0:
                pct = bytes_done * 100 // bytes_total
                logger.info(
                    "  done %s (%d%%, %s)",
                    format_elapsed(elapsed), pct, format_size(bytes_done),
                )
            else:
                logger.info("  done %s", format_elapsed(elapsed))
        # JSON timeline phase summary
        if emit_timeline:
            entry = {
                "t": round(time.monotonic() - t0, 3),
                "phase_done": phase,
                "duration_ms": round(elapsed * 1000),
                "cmds": cmds_done,
            }
            if bytes_total > 0:
                entry["bytes"] = bytes_done
            logger.debug("timeline: %s", json.dumps(entry, separators=(",", ":")))

    def _emit_timeline(self, cmd, phase: str, elapsed: float) -> None:
        """Emit per-command JSON timeline entry at DEBUG level."""
        entry = {
            "t": round(elapsed, 3),
            "phase": phase,
            "cmd": cmd.cmd_id,
            "op": cmd.op.name if hasattr(cmd.op, "name") else str(cmd.op),
        }
        if hasattr(cmd, "buffer_id") and cmd.buffer_id is not None:
            entry["buf"] = cmd.buffer_id
        if hasattr(cmd, "size") and cmd.size:
            entry["size"] = cmd.size
        logger.debug("timeline: %s", json.dumps(entry, separators=(",", ":")))

    # ── BO pre-allocation ──

    def _preallocate_bound_bos(
        self,
        commands: list[Command],
        tensor_data: dict[int, bytes],
    ) -> None:
        """Pre-allocate BOs for all auto_bind-referenced tensors."""
        bound_buffer_ids: set[int] = set()
        for binding in self._addr_bindings.values():
            bound_buffer_ids.add(binding[0])

        if not bound_buffer_ids:
            return

        buf_sizes: dict[int, int] = {}
        buf_ops: dict[int, OpCode] = {}
        for cmd in commands:
            if cmd.op in (OpCode.LOAD, OpCode.PULL):
                if cmd.buffer_id in bound_buffer_ids:
                    buf_sizes[cmd.buffer_id] = cmd.size or 4096
                    buf_ops[cmd.buffer_id] = cmd.op

        for buffer_id in sorted(bound_buffer_ids):
            if buffer_id in self._buffers:
                continue
            size = buf_sizes.get(buffer_id, 4096)
            mem_bank = self._mem_bank_map.get(buffer_id, 0)
            bo = self._xrt.bo(
                self._device, size,
                self._xrt.bo.flags.normal, mem_bank,
            )
            if hasattr(bo, "map_init"):
                bo.map_init()
            data = tensor_data.get(buffer_id, b"")
            if data:
                bo.write(data)
            else:
                bo.write(b"\x00" * size)
            bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._buffers[buffer_id] = bo
            addr = bo.address() if hasattr(bo, "address") else 0
            op_name = buf_ops.get(buffer_id, "?")
            if hasattr(op_name, "name"):
                op_name = op_name.name
            hex_head = " ".join(f"{b:02X}" for b in data[:32]) if data else "zeroed"
            logger.debug(
                "  pre-alloc BO: buf=%d size=%d bank=%d addr=0x%X (%s) [%s]",
                buffer_id, size, mem_bank, addr, op_name, hex_head,
            )

    # ── Dependency check ──

    def _wait_deps(self, cmd: Command) -> None:
        """Verify all dependencies are satisfied (host-side)."""
        for dep_id in cmd.dep:
            if dep_id not in self._completed:
                raise BackendError(
                    f"Dependency cmd_id={dep_id} not completed "
                    f"before cmd_id={cmd.cmd_id}"
                )

    # ── Command dispatch ──

    def _dispatch(self, cmd: Command, tensor_data: dict[int, bytes]) -> None:
        """Dispatch command to appropriate handler."""
        handler = {
            OpCode.LOAD: self._exec_load,
            OpCode.PUSH: self._exec_push,
            OpCode.PULL: self._exec_pull,
            OpCode.STORE: self._exec_store,
            OpCode.WRITE_REG: self._exec_write_reg,
            OpCode.READ_REG: self._exec_read_reg,
            OpCode.POLL_REG: self._exec_poll_reg,
            OpCode.BARRIER: self._exec_barrier,
            OpCode.COMPARE: self._exec_compare,
        }.get(cmd.op)

        if handler is None:
            raise BackendError(f"Unknown OpCode: {cmd.op}")

        if cmd.op == OpCode.LOAD:
            handler(cmd, tensor_data)
        else:
            handler(cmd)

    # ── Command handlers ──

    def _exec_load(self, cmd: Command, tensor_data: dict[int, bytes]) -> None:
        """LOAD: Host → Device Buffer."""
        data = tensor_data.get(cmd.buffer_id, b"")
        if not data:
            if cmd.buffer_id in self._buffers:
                return
            raise BackendError(
                f"LOAD cmd_id={cmd.cmd_id}: no tensor data for "
                f"buffer_id={cmd.buffer_id}"
            )
        bo = self._buffers.get(cmd.buffer_id)
        if bo is not None and hasattr(bo, "size") and bo.size() >= len(data):
            bo.write(data)
        else:
            mem_bank = self._mem_bank_map.get(cmd.buffer_id, 0)
            bo = self._xrt.bo(
                self._device, len(data),
                self._xrt.bo.flags.normal, mem_bank,
            )
            bo.write(data)
            self._buffers[cmd.buffer_id] = bo

    def _exec_push(self, cmd: Command) -> None:
        """PUSH: Sync BO to device."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            raise BackendError(
                f"PUSH cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                f"not loaded"
            )
        if cmd.buffer_id in self._prebound:
            return
        bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        arg_index = self._arg_map.get(cmd.buffer_id)
        if arg_index is not None:
            ip = self._get_ip(cmd.interface_id)
            ip.set_arg(arg_index, bo)

    def _exec_pull(self, cmd: Command) -> None:
        """PULL: Allocate output BO (actual sync deferred to STORE)."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            size = cmd.size or 4096
            mem_bank = self._mem_bank_map.get(cmd.buffer_id, 0)
            bo = self._xrt.bo(
                self._device, size,
                self._xrt.bo.flags.normal, mem_bank,
            )
            if hasattr(bo, "map_init"):
                bo.map_init()
            bo.write(b"\x00" * size)
            bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._buffers[cmd.buffer_id] = bo

    def _exec_store(self, cmd: Command) -> None:
        """STORE: Deferred until all commands complete."""
        self._deferred_stores.append(cmd)

    def _flush_deferred_stores(self) -> None:
        """Execute all deferred STORE commands."""
        for cmd in self._deferred_stores:
            bo = self._buffers.get(cmd.buffer_id)
            if bo is None:
                raise BackendError(
                    f"STORE cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                    f"not found"
                )
            bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            size = cmd.size if cmd.size > 0 else bo.size()
            if hasattr(bo, "map_read"):
                data = bo.map_read(size)
            else:
                data = bo.read(size)
            self._output_buffers[cmd.buffer_id] = bytes(data)
            hex_head = " ".join(f"{b:02X}" for b in data[:32])
            logger.info(
                "  buf=%d %s: [%s]",
                cmd.buffer_id, format_size(size), hex_head,
            )
        self._deferred_stores.clear()

    def _exec_write_reg(self, cmd: Command) -> None:
        """WRITE_REG: Write to IP control register."""
        ip = self._get_ip(cmd.interface_id)
        key = (cmd.interface_id, cmd.reg_offset)
        if key in self._addr_bindings:
            binding = self._addr_bindings[key]
            if len(binding) == 3:
                buffer_id, bits_spec, byte_offset = binding
            else:
                buffer_id, bits_spec = binding
                byte_offset = 0
            bo = self._buffers.get(buffer_id)
            if bo is not None and hasattr(bo, "address"):
                addr = bo.address() + byte_offset
                if bits_spec:
                    hi, lo = parse_bit_range(bits_spec)
                    addr = (addr >> lo) & ((1 << (hi - lo + 1)) - 1)
                logger.debug(
                    "  auto_bind: iface=%d reg=0x%X → buf=%d addr=0x%X",
                    cmd.interface_id, cmd.reg_offset, buffer_id, addr,
                )
                ip.write_register(cmd.reg_offset, addr)
                return
            else:
                logger.warning(
                    "  auto_bind MISS: iface=%d reg=0x%X buf=%d → fallback val=0x%X",
                    cmd.interface_id, cmd.reg_offset, buffer_id, cmd.reg_value,
                )
        logger.debug(
            "  WRITE_REG iface=%d reg=0x%X val=0x%X",
            cmd.interface_id, cmd.reg_offset, cmd.reg_value,
        )
        ip.write_register(cmd.reg_offset, cmd.reg_value)

    def _exec_read_reg(self, cmd: Command) -> None:
        """READ_REG: Read IP control register."""
        ip = self._get_ip(cmd.interface_id)
        cmd.reg_value = ip.read_register(cmd.reg_offset)

    def _exec_poll_reg(self, cmd: Command) -> None:
        """POLL_REG: Poll register with progress reporting and stall detection.

        Reports progress every 2s (only after quiet period).
        Emits WARNING after 10s of no value change.
        """
        ip = self._get_ip(cmd.interface_id)
        start = time.monotonic()
        poll_count = 0
        last_report_time = start
        last_val: int | None = None
        last_change_time = start

        while True:
            val = ip.read_register(cmd.reg_offset)
            poll_count += 1
            now = time.monotonic()
            elapsed = now - start

            if (val & cmd.reg_mask) == cmd.reg_expected:
                logger.info(
                    "  POLL (==0x%X): %d polls, %s",
                    cmd.reg_expected, poll_count, format_elapsed(elapsed),
                )
                cmd.reg_value = val
                return

            # Track value changes for stall detection
            if last_val is None or val != last_val:
                last_val = val
                last_change_time = now

            # Progress reports (only after quiet period)
            if elapsed >= _QUIET_PERIOD and (now - last_report_time) >= _POLL_REPORT_INTERVAL:
                logger.info(
                    "  [%s] POLL 0x%X: val=0x%X (%d polls)",
                    format_elapsed(elapsed), cmd.reg_offset, val, poll_count,
                )
                last_report_time = now

            # Stall detection
            no_change = now - last_change_time
            if no_change >= _STALL_THRESHOLD:
                logger.warning(
                    "  [%s] STALL POLL_REG: no change %.0fs "
                    "(addr=0x%X mask=0x%X expected=0x%X val=0x%X)",
                    format_elapsed(elapsed), no_change,
                    cmd.reg_offset, cmd.reg_mask, cmd.reg_expected, val,
                )
                last_change_time = now  # reset to avoid spam

            elapsed_ms = elapsed * 1000
            if elapsed_ms > self._poll_timeout_ms:
                raise PollTimeoutError(
                    f"POLL_REG timeout at offset 0x{cmd.reg_offset:X} "
                    f"after {self._poll_timeout_ms}ms "
                    f"(last value=0x{val:X}, mask=0x{cmd.reg_mask:X}, "
                    f"expected=0x{cmd.reg_expected:X})"
                )
            time.sleep(0.001)  # 1ms between polls

    def _exec_barrier(self, cmd: Command) -> None:
        """BARRIER: Host-side fence — all prior commands already complete."""
        pass

    def _exec_compare(self, cmd: Command) -> None:
        """COMPARE: Host-side buffer comparison."""
        pass

    # ── Cleanup ──

    def cleanup(self) -> None:
        """Release XRT buffer objects."""
        self._buffers.clear()
        self._output_buffers.clear()
        self._completed.clear()
        self._prebound.clear()
        self._deferred_stores.clear()
