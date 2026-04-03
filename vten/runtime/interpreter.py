"""CommandInterpreter — IR Command to XRT API translator.

The HW counterpart to SHM Packer (Stage 7). While SHM Packer serializes
commands into a binary image for the hardware scheduler, CommandInterpreter
executes them sequentially on the host via XRT API calls.

Uses xrt.ip (raw register access) for maximum compatibility with RTL designs.

Spec reference: 08_backend_abstraction.md §4.3
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from vten.errors import BackendError, PollTimeoutError
from vten.runtime.binder import parse_bit_range
from vten.spec.models import OpCode, Protocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.runtime.ir import Command


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
        """Initialize interpreter with XRT resources.

        Args:
            device: pyxrt.device instance.
            kernel: Default pyxrt.ip (or pyxrt.kernel) instance.
            xrt_module: The pyxrt module (for bo/sync constants).
            arg_map: buffer_id → kernel arg_index mapping.
            poll_timeout_ms: Timeout for POLL_REG operations.
            ip_map: interface_id → pyxrt.ip mapping for multi-IP.
            mem_bank_map: buffer_id → memory bank index mapping.
            addr_bindings: (interface_id, reg_offset) → (buffer_id, bits)
                mapping for auto_bind address substitution.
        """
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

    def execute(
        self,
        commands: list[Command],
        tensor_data: dict[int, bytes],
    ) -> None:
        """Execute IR commands sequentially.

        Args:
            commands: Ordered list of IR Commands from Stage 6.
            tensor_data: buffer_id → serialized tensor bytes.
        """
        self._output_buffers.clear()
        self._completed.clear()

        # Pre-allocate all BOs referenced by auto_bind address registers.
        # WRITE_REG (configure) runs before LOAD/PUSH (send) and PULL (recv),
        # so BOs must exist before any command executes.
        self._preallocate_bound_bos(commands, tensor_data)

        # Collect (iface_id, reg_offset) pairs covered by WRITE_REG commands.
        # These are already handled by _exec_write_reg with auto_bind substitution.
        covered_regs: set[tuple[int, int]] = set()
        for cmd in commands:
            if cmd.op == OpCode.WRITE_REG:
                covered_regs.add((cmd.interface_id, cmd.reg_offset))

        t0 = time.monotonic()
        addr_written = False
        for i, cmd in enumerate(commands):
            # Write uncovered auto_bind address registers after the last
            # configure WRITE_REG but before data transfer / VSYNC.
            if not addr_written and cmd.op != OpCode.WRITE_REG:
                self._write_addr_bindings(covered_regs)
                addr_written = True
            logger.info(
                "  [%d/%d] cmd_id=%d %s buf=%s iface=%s",
                i + 1, len(commands), cmd.cmd_id,
                cmd.op.name if hasattr(cmd.op, "name") else cmd.op,
                getattr(cmd, "buffer_id", "-"),
                getattr(cmd, "interface_id", "-"),
            )
            self._wait_deps(cmd)
            self._dispatch(cmd, tensor_data)
            elapsed = time.monotonic() - t0
            logger.info("    done (%.1fs elapsed)", elapsed)
            self._completed.add(cmd.cmd_id)

        # Flush any remaining deferred stores (safety net)
        if self._deferred_stores:
            self._flush_deferred_stores()

    def _write_addr_bindings(
        self, covered_regs: set[tuple[int, int]] | None = None,
    ) -> None:
        """Write uncovered auto_bind address registers to their IPs.

        Only writes registers NOT already handled by WRITE_REG commands.
        This ensures per-port address registers (e.g., per-bank weight
        addresses 1-31) are written without double-writing the ones
        that the configure step handles via auto_bind substitution.
        """
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
            logger.info(
                "  addr_bindings: wrote %d, skipped %d (covered by WRITE_REG)",
                written, skipped,
            )

    def _preallocate_bound_bos(
        self,
        commands: list[Command],
        tensor_data: dict[int, bytes],
    ) -> None:
        """Pre-allocate BOs for all auto_bind-referenced tensors.

        In XRT, address registers (auto_bind) must contain the BO device
        address. WRITE_REG (configure) runs before LOAD/PUSH (send) and
        PULL (recv), so the BOs don't exist yet. Pre-allocate them here
        so auto_bind substitution resolves to real device addresses.

        For input BOs (LOAD), also writes tensor data and syncs TO_DEVICE.
        For output BOs (PULL), allocates empty and syncs TO_DEVICE to
        force hw_emu address assignment.
        """
        # Collect buffer_ids referenced by addr_bindings
        bound_buffer_ids: set[int] = set()
        for binding in self._addr_bindings.values():
            bound_buffer_ids.add(binding[0])

        if not bound_buffer_ids:
            return

        # Collect size info from commands
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
            # For input tensors, write data into BO
            data = tensor_data.get(buffer_id, b"")
            if data:
                bo.write(data)
            # Sync TO_DEVICE to assign device address and transfer data
            bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self._buffers[buffer_id] = bo
            addr = bo.address() if hasattr(bo, "address") else 0
            op_name = buf_ops.get(buffer_id, "?")
            if hasattr(op_name, "name"):
                op_name = op_name.name
            hex_head = " ".join(f"{b:02X}" for b in data[:32]) if data else "-"
            logger.info(
                "  pre-allocated BO: buffer_id=%d size=%d bank=%d "
                "addr=0x%X (%s, data=%d bytes) first32=[%s]",
                buffer_id, size, mem_bank, addr, op_name, len(data),
                hex_head,
            )

    def _wait_deps(self, cmd: Command) -> None:
        """Verify all dependencies are satisfied (host-side)."""
        for dep_id in cmd.dep:
            if dep_id not in self._completed:
                raise BackendError(
                    f"Dependency cmd_id={dep_id} not completed "
                    f"before cmd_id={cmd.cmd_id}"
                )

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

    def _exec_load(self, cmd: Command, tensor_data: dict[int, bytes]) -> None:
        """LOAD: Host → Device Buffer. Create BO and write tensor data.

        If the BO was pre-allocated (for auto_bind address resolution),
        reuses it instead of creating a new one on potentially wrong bank.
        Inference: if BO exists and no data → skip (prebound device buffer).
        """
        data = tensor_data.get(cmd.buffer_id, b"")
        if not data:
            if cmd.buffer_id in self._buffers:
                return  # BO already on device (prebound) → skip
            raise BackendError(
                f"LOAD cmd_id={cmd.cmd_id}: no tensor data for "
                f"buffer_id={cmd.buffer_id}"
            )
        bo = self._buffers.get(cmd.buffer_id)
        if bo is not None and hasattr(bo, "size") and bo.size() >= len(data):
            bo.write(data)  # Reuse existing BO
        else:
            mem_bank = self._mem_bank_map.get(cmd.buffer_id, 0)
            bo = self._xrt.bo(
                self._device, len(data),
                self._xrt.bo.flags.normal, mem_bank,
            )
            bo.write(data)
            self._buffers[cmd.buffer_id] = bo

    def _exec_push(self, cmd: Command) -> None:
        """PUSH: Sync BO to device. Skip if prebound (already synced)."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            raise BackendError(
                f"PUSH cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                f"not loaded"
            )
        if cmd.buffer_id in self._prebound:
            return  # Already synced to device
        bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        arg_index = self._arg_map.get(cmd.buffer_id)
        if arg_index is not None:
            ip = self._get_ip(cmd.interface_id)
            ip.set_arg(arg_index, bo)

    def _exec_pull(self, cmd: Command) -> None:
        """PULL: Prepare output BO for device-to-host transfer.

        Unlike SIM where PULL activates a BFM slave entry, XRT PULL
        only allocates the output buffer. The actual sync FROM device
        is deferred to STORE, after the kernel writes data.
        """
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            mem_bank = self._mem_bank_map.get(cmd.buffer_id, 0)
            bo = self._xrt.bo(
                self._device, cmd.size or 4096,
                self._xrt.bo.flags.normal, mem_bank,
            )
            self._buffers[cmd.buffer_id] = bo

    def _exec_store(self, cmd: Command) -> None:
        """STORE: Sync BO from device, then read data back to host.

        In XRT, STORE must happen after the DUT finishes computation.
        If STORE appears before POLL_REG in the command list (because
        recv_tensor is called before poll_register for xsim BFM compat),
        defer execution until after POLL_REG completes.
        """
        self._deferred_stores.append(cmd)

    def _flush_deferred_stores(self) -> None:
        """Execute all deferred STORE commands (after POLL_REG)."""
        for cmd in self._deferred_stores:
            bo = self._buffers.get(cmd.buffer_id)
            if bo is None:
                raise BackendError(
                    f"STORE cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                    f"not found"
                )
            bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            size = cmd.size if cmd.size > 0 else bo.size()
            data = bo.read(size)
            self._output_buffers[cmd.buffer_id] = bytes(data)
            hex_head = " ".join(f"{b:02X}" for b in data[:32])
            logger.info(
                "    deferred STORE: buf=%d size=%d first32=[%s]",
                cmd.buffer_id, size, hex_head,
            )
        self._deferred_stores.clear()

    def _exec_write_reg(self, cmd: Command) -> None:
        """WRITE_REG: Write to IP control register.

        If this register has an auto_bind address binding, substitutes
        the SHM offset with the actual BO device address + byte offset.
        """
        ip = self._get_ip(cmd.interface_id)
        key = (cmd.interface_id, cmd.reg_offset)
        if key in self._addr_bindings:
            binding = self._addr_bindings[key]
            # Support both (buffer_id, bits) and (buffer_id, bits, offset)
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
                logger.info(
                    "    auto_bind: iface=%d reg=0x%X → buf=%d "
                    "bo_addr=0x%X offset=%d bits=%s → val=0x%X",
                    cmd.interface_id, cmd.reg_offset, buffer_id,
                    bo.address(), byte_offset, bits_spec, addr,
                )
                ip.write_register(cmd.reg_offset, addr)
                return
            else:
                logger.warning(
                    "    auto_bind MISS: iface=%d reg=0x%X buf=%d "
                    "bo=%s → fallback val=0x%X",
                    cmd.interface_id, cmd.reg_offset, buffer_id,
                    "None" if bo is None else "no .address()",
                    cmd.reg_value,
                )
        ip.write_register(cmd.reg_offset, cmd.reg_value)

    def _exec_read_reg(self, cmd: Command) -> None:
        """READ_REG: Read IP control register."""
        ip = self._get_ip(cmd.interface_id)
        cmd.reg_value = ip.read_register(cmd.reg_offset)

    def _exec_poll_reg(self, cmd: Command) -> None:
        """POLL_REG: Poll register until (value & mask) == expected."""
        ip = self._get_ip(cmd.interface_id)
        start = time.monotonic()
        poll_count = 0
        while True:
            val = ip.read_register(cmd.reg_offset)
            poll_count += 1
            if (val & cmd.reg_mask) == cmd.reg_expected:
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.info(
                    "    POLL_REG done: 0x%X=%d polls, %.1fs",
                    cmd.reg_offset, poll_count, elapsed_ms / 1000,
                )
                cmd.reg_value = val
                # Flush STORE commands deferred from before this POLL
                if self._deferred_stores:
                    self._flush_deferred_stores()
                return
            elapsed_ms = (time.monotonic() - start) * 1000
            if poll_count % 1000 == 0:
                logger.info(
                    "    POLL_REG 0x%X: %d polls, %.1fs, val=0x%X",
                    cmd.reg_offset, poll_count, elapsed_ms / 1000, val,
                )
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

    def cleanup(self) -> None:
        """Release XRT buffer objects."""
        self._buffers.clear()
        self._output_buffers.clear()
        self._completed.clear()
        self._prebound.clear()
        self._deferred_stores.clear()
