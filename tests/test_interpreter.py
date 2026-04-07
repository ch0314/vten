"""Phase B tests: CommandInterpreter — OpCode dispatch, dep tracking, error handling.

Spec reference: 08_backend_abstraction.md §4.3, §6.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from vten.spec.models import OpCode, Protocol


# ── Mock XRT module ──


class MockBO:
    """Mock XRT buffer object."""

    def __init__(self, device, size, flags, mem_group):
        self._data = bytearray(size)
        self._size = size

    def write(self, data):
        self._data[:len(data)] = data

    def read(self, size):
        return bytes(self._data[:size])

    def sync(self, direction):
        pass

    def size(self):
        return self._size


class MockBOFlags:
    normal = 0


class MockSyncDirection:
    XCL_BO_SYNC_BO_TO_DEVICE = 0
    XCL_BO_SYNC_BO_FROM_DEVICE = 1


def _mock_xrt():
    """Create mock pyxrt module."""
    xrt = MagicMock()
    xrt.bo = MockBO
    xrt.bo.flags = MockBOFlags()
    xrt.xclBOSyncDirection = MockSyncDirection()
    return xrt


def _mock_kernel():
    """Create mock XRT kernel."""
    kernel = MagicMock()
    kernel.write_register = MagicMock()
    kernel.read_register = MagicMock(return_value=0)
    kernel.set_arg = MagicMock()
    return kernel


def _make_cmd(op, cmd_id=0, buffer_id=0, size=0, dep=None, **kwargs):
    """Create IR Command for testing."""
    from vten.runtime.ir import Command
    return Command(
        op=op,
        cmd_id=cmd_id,
        buffer_id=buffer_id,
        size=size,
        dep=dep or [],
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# §1  Basic dispatch
# ═══════════════════════════════════════════════════════════════════


class TestCommandInterpreterDispatch:
    """CommandInterpreter dispatches OpCodes to handlers."""

    def test_load_creates_buffer(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=64)
        tensor_data = {1: b"\x00" * 64}
        interp.execute([cmd], tensor_data)
        assert 0 in interp._completed

    def test_load_missing_data_raises(self):
        from vten.errors import BackendError
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=99)
        with pytest.raises(BackendError, match="no tensor data"):
            interp.execute([cmd], {})

    def test_store_populates_output_buffers(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        load_cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=32)
        store_cmd = _make_cmd(OpCode.STORE, cmd_id=1, buffer_id=1, size=32, dep=[0])
        interp.execute([load_cmd, store_cmd], {1: b"\xAB" * 32})
        assert 1 in interp.output_buffers
        assert len(interp.output_buffers[1]) == 32

    def test_store_missing_buffer_raises(self):
        from vten.errors import BackendError
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.STORE, cmd_id=0, buffer_id=99, size=32)
        with pytest.raises(BackendError, match="not found"):
            interp.execute([cmd], {})

    def test_write_reg_calls_kernel(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.WRITE_REG, cmd_id=0, reg_offset=0x10, reg_value=0xFF)
        interp.execute([cmd], {})
        kernel.write_register.assert_called_once_with(0x10, 0xFF)

    def test_read_reg_calls_kernel(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        kernel.read_register.return_value = 42
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.READ_REG, cmd_id=0, reg_offset=0x10)
        interp.execute([cmd], {})
        kernel.read_register.assert_called_once_with(0x10)
        assert cmd.reg_value == 42

    def test_poll_reg_succeeds_immediately(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        kernel.read_register.return_value = 0x01
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.POLL_REG, cmd_id=0, reg_offset=0x10,
                        reg_mask=0x01, reg_expected=0x01)
        interp.execute([cmd], {})
        assert cmd.reg_value == 0x01

    def test_poll_reg_timeout_raises(self):
        from vten.errors import PollTimeoutError
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        kernel.read_register.return_value = 0x00  # never matches
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
            poll_timeout_ms=10,  # very short timeout
        )
        cmd = _make_cmd(OpCode.POLL_REG, cmd_id=0, reg_offset=0x10,
                        reg_mask=0x01, reg_expected=0x01)
        with pytest.raises(PollTimeoutError):
            interp.execute([cmd], {})

    def test_barrier_is_noop(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.BARRIER, cmd_id=0)
        interp.execute([cmd], {})
        assert 0 in interp._completed

    def test_push_syncs_and_sets_arg(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
            arg_map={1: 0},
        )
        load_cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=16)
        push_cmd = _make_cmd(OpCode.PUSH, cmd_id=1, buffer_id=1, dep=[0])
        interp.execute([load_cmd, push_cmd], {1: b"\x00" * 16})
        kernel.set_arg.assert_called_once()

    def test_push_missing_buffer_raises(self):
        from vten.errors import BackendError
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.PUSH, cmd_id=0, buffer_id=99)
        with pytest.raises(BackendError, match="not loaded"):
            interp.execute([cmd], {})


# ═══════════════════════════════════════════════════════════════════
# §2  Dependency tracking
# ═══════════════════════════════════════════════════════════════════


class TestCommandInterpreterDeps:
    """Dependency validation in CommandInterpreter."""

    def test_unsatisfied_dep_raises(self):
        from vten.errors import BackendError
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.BARRIER, cmd_id=1, dep=[999])
        with pytest.raises(BackendError, match="not completed"):
            interp.execute([cmd], {})

    def test_sequential_deps_work(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmds = [
            _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=8),
            _make_cmd(OpCode.PUSH, cmd_id=1, buffer_id=1, dep=[0]),
            _make_cmd(OpCode.BARRIER, cmd_id=2, dep=[1]),
        ]
        interp.execute(cmds, {1: b"\x00" * 8})
        assert interp._completed == {0, 1, 2}

    def test_cleanup_clears_state(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=8)
        interp.execute([cmd], {1: b"\x00" * 8})
        assert len(interp._buffers) > 0
        interp.cleanup()
        assert len(interp._buffers) == 0
        assert len(interp._output_buffers) == 0


# ═══════════════════════════════════════════════════════════════════
# §3  Full pipeline
# ═══════════════════════════════════════════════════════════════════


class TestCommandInterpreterPipeline:
    """Full LOAD → PUSH → PULL → STORE pipeline."""

    def test_full_data_path(self):
        from vten.backend.xrt_interpreter import CommandInterpreter

        kernel = _mock_kernel()
        interp = CommandInterpreter(
            device=MagicMock(), kernel=kernel,
            xrt_module=_mock_xrt(),
            arg_map={1: 0, 2: 1},
        )
        data = b"\xDE\xAD" * 16
        cmds = [
            _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=len(data)),
            _make_cmd(OpCode.PUSH, cmd_id=1, buffer_id=1, dep=[0]),
            _make_cmd(OpCode.WRITE_REG, cmd_id=2, reg_offset=0x00, reg_value=1, dep=[1]),
            _make_cmd(OpCode.PULL, cmd_id=3, buffer_id=2, size=len(data), dep=[2]),
            _make_cmd(OpCode.STORE, cmd_id=4, buffer_id=2, size=len(data), dep=[3]),
        ]
        interp.execute(cmds, {1: data})

        assert 2 in interp.output_buffers
        assert len(interp.output_buffers[2]) == len(data)
        assert interp._completed == {0, 1, 2, 3, 4}

    def test_unknown_opcode_raises(self):
        from vten.errors import BackendError
        from vten.backend.xrt_interpreter import CommandInterpreter

        interp = CommandInterpreter(
            device=MagicMock(), kernel=_mock_kernel(),
            xrt_module=_mock_xrt(),
        )
        cmd = _make_cmd(OpCode.LOAD, cmd_id=0, buffer_id=1, size=8)
        cmd.op = 99  # Invalid opcode
        with pytest.raises(BackendError, match="Unknown OpCode"):
            interp.execute([cmd], {1: b"\x00" * 8})
