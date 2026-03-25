"""CommandInterpreter — IR Command to XRT API translator.

The HW counterpart to SHM Packer (Stage 7). While SHM Packer serializes
commands into a binary image for the hardware scheduler, CommandInterpreter
executes them sequentially on the host via XRT API calls.

Spec reference: 08_backend_abstraction.md §4.3
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vten.errors import BackendError, PollTimeoutError
from vten.spec.models import OpCode, Protocol

if TYPE_CHECKING:
    from vten.runtime.ir import Command


class CommandInterpreter:
    """Interprets IR Commands and executes them via XRT API.

    OpCode mapping:
      LOAD      → BO.write(data) — host writes tensor data into device buffer
      PUSH      → BO.sync(TO_DEVICE) + kernel.set_arg() — transfer to FPGA
      PULL      → BO.sync(FROM_DEVICE) — transfer from FPGA
      STORE     → BO.read(size) — host reads device buffer
      WRITE_REG → kernel.write_register(offset, value)
      READ_REG  → kernel.read_register(offset)
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
    ) -> None:
        """Initialize interpreter with XRT resources.

        Args:
            device: pyxrt.device instance.
            kernel: pyxrt.kernel instance.
            xrt_module: The pyxrt module (for bo/sync constants).
            arg_map: buffer_id → kernel arg_index mapping.
            poll_timeout_ms: Timeout for POLL_REG operations.
        """
        self._device = device
        self._kernel = kernel
        self._xrt = xrt_module
        self._arg_map = arg_map or {}
        self._poll_timeout_ms = poll_timeout_ms
        self._buffers: dict[int, Any] = {}  # buffer_id → XRT BO
        self._output_buffers: dict[int, bytes] = {}
        self._completed: set[int] = set()  # completed cmd_ids

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

        for cmd in commands:
            self._wait_deps(cmd)
            self._dispatch(cmd, tensor_data)
            self._completed.add(cmd.cmd_id)

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
        """LOAD: Host → Device Buffer. Create BO and write tensor data."""
        data = tensor_data.get(cmd.buffer_id, b"")
        if not data:
            raise BackendError(
                f"LOAD cmd_id={cmd.cmd_id}: no tensor data for "
                f"buffer_id={cmd.buffer_id}"
            )
        bo = self._xrt.bo(
            self._device, len(data),
            self._xrt.bo.flags.normal, 0,
        )
        bo.write(data)
        self._buffers[cmd.buffer_id] = bo

    def _exec_push(self, cmd: Command) -> None:
        """PUSH: Sync BO to device + set kernel arg."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            raise BackendError(
                f"PUSH cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                f"not loaded"
            )
        bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        arg_index = self._arg_map.get(cmd.buffer_id)
        if arg_index is not None:
            self._kernel.set_arg(arg_index, bo)

    def _exec_pull(self, cmd: Command) -> None:
        """PULL: Sync BO from device."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            # Create output BO for pull
            bo = self._xrt.bo(
                self._device, cmd.size or 4096,
                self._xrt.bo.flags.normal, 0,
            )
            self._buffers[cmd.buffer_id] = bo

            arg_index = self._arg_map.get(cmd.buffer_id)
            if arg_index is not None:
                self._kernel.set_arg(arg_index, bo)

        bo.sync(self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    def _exec_store(self, cmd: Command) -> None:
        """STORE: Read BO data back to host."""
        bo = self._buffers.get(cmd.buffer_id)
        if bo is None:
            raise BackendError(
                f"STORE cmd_id={cmd.cmd_id}: buffer_id={cmd.buffer_id} "
                f"not found"
            )
        size = cmd.size if cmd.size > 0 else bo.size()
        data = bo.read(size)
        self._output_buffers[cmd.buffer_id] = bytes(data)

    def _exec_write_reg(self, cmd: Command) -> None:
        """WRITE_REG: Write to kernel control register."""
        self._kernel.write_register(cmd.reg_offset, cmd.reg_value)

    def _exec_read_reg(self, cmd: Command) -> None:
        """READ_REG: Read kernel control register."""
        cmd.reg_value = self._kernel.read_register(cmd.reg_offset)

    def _exec_poll_reg(self, cmd: Command) -> None:
        """POLL_REG: Poll register until (value & mask) == expected."""
        start = time.monotonic()
        while True:
            val = self._kernel.read_register(cmd.reg_offset)
            if (val & cmd.reg_mask) == cmd.reg_expected:
                cmd.reg_value = val
                return
            elapsed_ms = (time.monotonic() - start) * 1000
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
