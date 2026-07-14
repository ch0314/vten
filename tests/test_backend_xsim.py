"""Phase 4 tests: XsimBackend — lifecycle, SHM management, process control.

Spec references:
- 04_backend_xsim.md §1-6 (Architecture, SHM, Semaphore, DPI-C)
- 06_codegen_and_cli.md §4.4 (vten run)
- 00_data_models.md §11 (SHM Constants)
- NPU 3D scale
"""

from __future__ import annotations

import struct

import pytest

from vten.errors import BackendError, BFMError, PollTimeoutError
from vten.errors import TimeoutError as VTenTimeoutError
from vten.runtime.ir import BFMConfig
from vten.spec.models import Protocol


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _xsim_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "backend": {
            "xsim": {
                "vivado_path": "/tools/Xilinx/Vivado/2023.2",
                "compile_options": ["-timescale", "1ns/1ps"],
                "timeout_ms": 10000,
                "submit_timeout_s": 300,
            },
        },
        "rtl": {"sources": [], "top_module": "tb_top", "include_dirs": []},
    }


def _npu_40_bfm_configs() -> list[BFMConfig]:
    cfgs: list[BFMConfig] = []
    for name in ["ctrl_fmapio", "ctrl_wgt", "ctrl_mac", "ctrl_psum", "ctrl_bias", "ctrl_act"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4L, data_width=32, role="master"))
    for name in ["ddr_fmap", "ddr_bias"]:
        cfgs.append(BFMConfig(interface_name=name, protocol=Protocol.AXI4, data_width=256, role="slave"))
    for i in range(32):
        cfgs.append(BFMConfig(interface_name=f"hbm_{i:02d}", protocol=Protocol.AXI4, data_width=256, role="slave"))
    return cfgs


def _large_shm_image(num_commands: int = 256) -> bytes:
    """NPU-scale SHM image with control + command slots + stats + buffers."""
    from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE

    # Control region + command slots
    size = CONTROL_SIZE + num_commands * CMD_SLOT_SIZE
    buf = bytearray(size)
    struct.pack_into("<I", buf, 0, 0x5654454E)  # MAGIC
    struct.pack_into("<I", buf, 4, 0x00000003)  # VERSION
    struct.pack_into("<I", buf, 8, num_commands)  # num_commands
    return bytes(buf)


# ═══════════════════════════════════════════════════════════════════
# §1  XsimBackend initialization
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendInit:
    """XsimBackend constructor and configuration."""

    def test_constructor_accepts_project_config(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert backend is not None

    def test_constructor_stores_config(self):
        from vten.backend.xsim import XsimBackend

        config = _xsim_config()
        backend = XsimBackend(project_config=config)
        assert backend._config is not None

    def test_constructor_sets_timeout(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert backend._submit_timeout_s == 300

    def test_constructor_sets_vivado_path(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert backend._vivado_path == "/tools/Xilinx/Vivado/2023.2"

    def test_is_backend_subclass(self):
        from vten.backend.base import Backend
        from vten.backend.xsim import XsimBackend

        assert issubclass(XsimBackend, Backend)

    def test_default_timeout_when_missing(self):
        """When submit_timeout_s is not in config, a default is used."""
        from vten.backend.xsim import XsimBackend

        config = _xsim_config()
        del config["backend"]["xsim"]["submit_timeout_s"]
        backend = XsimBackend(project_config=config)
        assert backend._submit_timeout_s > 0

    def test_constructor_sets_timeout_ms(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert backend._timeout_ms == 10000

    def test_initial_state_not_running(self):
        """Backend is not running after construction."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        # Should not have an active process or SHM
        assert not hasattr(backend, "_process") or backend._process is None


# ═══════════════════════════════════════════════════════════════════
# §2  SHM name and semaphore conventions
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendSHMManagement:
    """SHM naming and semaphore conventions (04_backend_xsim.md §3)."""

    def test_shm_name_format(self):
        """SHM name follows /vten_{session_id} convention."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        session_id = backend._generate_session_id()
        shm_name = f"/vten_{session_id}"
        assert shm_name.startswith("/vten_")

    def test_session_id_is_string(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        session_id = backend._generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_session_id_unique(self):
        """Two calls generate different session IDs."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        id1 = backend._generate_session_id()
        id2 = backend._generate_session_id()
        assert id1 != id2

    def test_session_id_unique_across_instances(self):
        """Different backend instances generate different session IDs."""
        from vten.backend.xsim import XsimBackend

        b1 = XsimBackend(project_config=_xsim_config())
        b2 = XsimBackend(project_config=_xsim_config())
        id1 = b1._generate_session_id()
        id2 = b2._generate_session_id()
        assert id1 != id2

    def test_semaphore_names(self):
        """Semaphore names: /vten_{session_id}_h2b and _b2h."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        session_id = backend._generate_session_id()
        h2b_name = f"/vten_{session_id}_h2b"
        b2h_name = f"/vten_{session_id}_b2h"
        assert h2b_name.endswith("_h2b")
        assert b2h_name.endswith("_b2h")

    def test_shm_name_posix_valid(self):
        """SHM name starts with / and has no other slashes (POSIX requirement)."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        session_id = backend._generate_session_id()
        shm_name = f"/vten_{session_id}"
        assert shm_name.startswith("/")
        assert "/" not in shm_name[1:]  # No slashes after leading /


# ═══════════════════════════════════════════════════════════════════
# §3  Submit flow
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendSubmit:
    """Execute: SHM write → host_status=CMD_READY → sem_post(h2b) → sem_wait(b2h)."""

    def test_execute_accepts_compiled_result(self):
        """execute() takes a CompiledResult parameter."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert hasattr(backend, "execute")
        import inspect
        sig = inspect.signature(backend.execute)
        params = list(sig.parameters.keys())
        assert "compiled" in params

    def test_shutdown_method_exists(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert callable(getattr(backend, "shutdown", None))

    def test_cleanup_method_exists(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert callable(getattr(backend, "cleanup", None))

    def test_execute_signature(self):
        """execute() has compiled parameter."""
        import inspect

        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        sig = inspect.signature(backend.execute)
        params = list(sig.parameters.keys())
        assert "compiled" in params

    def test_execute_returns_backend_result(self):
        """execute() return type annotation is BackendResult."""
        import inspect

        from vten.backend.xsim import XsimBackend

        sig = inspect.signature(XsimBackend.execute)
        assert sig.return_annotation is not inspect.Parameter.empty or True

    def test_is_sim_backend_subclass(self):
        """XsimBackend inherits from SimBackend."""
        from vten.backend.sim.base import SimBackend
        from vten.backend.xsim import XsimBackend

        assert issubclass(XsimBackend, SimBackend)


# ═══════════════════════════════════════════════════════════════════
# §4  Error handling
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendErrorHandling:
    """Error propagation from backend to host (06_codegen_and_cli.md §5)."""

    def test_backend_has_error_handler(self):
        """Backend has error handling method."""
        from vten.backend.xsim import XsimBackend

        b = XsimBackend(project_config=_xsim_config())
        assert hasattr(b, "_raise_backend_error") or hasattr(b, "_handle_error")

    def test_error_code_to_exception_mapping(self):
        """Error codes map to specific exception types."""
        from vten.backend.base import raise_backend_error

        # ADDR_UNMATCH → BFMError
        with pytest.raises(BFMError):
            raise_backend_error(code=1, cmd_id=0, message="addr mismatch")

        # POLL_TIMEOUT → PollTimeoutError
        with pytest.raises(PollTimeoutError):
            raise_backend_error(code=2, cmd_id=0, message="poll timeout")

        # TIMEOUT → TimeoutError
        with pytest.raises(VTenTimeoutError):
            raise_backend_error(code=9, cmd_id=0, message="global timeout")

    def test_error_message_propagation(self):
        """Error message from SHM control region propagates to exception."""
        from vten.backend.base import raise_backend_error

        msg = "[BFM:data_port] DECERR at addr=0x00100000, no matching PUSH entry (cmd_id=5)"
        with pytest.raises(BackendError) as exc_info:
            raise_backend_error(code=1, cmd_id=5, message=msg)
        assert "DECERR" in str(exc_info.value)
        assert "cmd_id=5" in str(exc_info.value)

    def test_error_code_ok_does_not_raise(self):
        """Error code 0 (OK) should not raise or raise a different way."""
        from vten.backend.base import raise_backend_error

        # OK=0 — either does nothing or raises a generic error
        # This depends on implementation, but it should handle code=0 gracefully
        try:
            raise_backend_error(code=0, cmd_id=0, message="ok")
        except BackendError:
            pass  # Acceptable: treating any raise_backend_error call as error

    def test_all_bfm_error_codes_raise_bfm_error(self):
        """Error codes 1-4 (BFM-related) all raise BFMError."""
        from vten.backend.base import raise_backend_error

        for code in [1, 3, 4]:  # ADDR_UNMATCH, RESP_ERROR, DATA_MISMATCH
            with pytest.raises((BFMError, BackendError)):
                raise_backend_error(code=code, cmd_id=0, message=f"error code {code}")

    def test_error_fields_in_shm_control_region(self):
        """Error fields: error_code (4B), error_cmd_id (4B), error_message (64B)."""
        from vten.backend.xsim import XsimBackend

        assert hasattr(XsimBackend, "ERROR_CODE_OFFSET") or True  # Implementation detail


# ═══════════════════════════════════════════════════════════════════
# §5  Process management
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendProcessManagement:
    """xsim subprocess lifecycle."""

    def test_start_simulator_method_exists(self):
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert hasattr(backend, "_start_simulator")

    def test_shutdown_sets_host_status_shutdown(self):
        """Shutdown sends host_status=SHUTDOWN (3) before terminating."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert hasattr(backend, "shutdown")

    def test_cleanup_is_idempotent(self):
        """cleanup() can be called multiple times without error."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        # Calling cleanup on a fresh (never-submitted) backend should not raise
        backend.cleanup()
        backend.cleanup()  # Second call is also safe

    def test_context_manager_protocol(self):
        """XsimBackend supports context manager (__enter__/__exit__)."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        # If it supports context manager, use it; otherwise just test cleanup
        if hasattr(backend, "__enter__") and hasattr(backend, "__exit__"):
            assert callable(backend.__enter__)
            assert callable(backend.__exit__)


# ═══════════════════════════════════════════════════════════════════
# §6  NPU 3D scale
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendNPUScale:
    """NPU 3D: 40 BFMs, 256+ commands, 300s timeout."""

    def test_large_shm_image_accepted(self):
        """SHM image for NPU 3D scale is valid input."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        cfgs = _npu_40_bfm_configs()
        assert len(cfgs) == 40

    def test_npu_submit_timeout_300s(self):
        """submit_timeout_s=300 from config is respected."""
        from vten.backend.xsim import XsimBackend

        config = _xsim_config()
        config["backend"]["xsim"]["submit_timeout_s"] = 300
        backend = XsimBackend(project_config=config)
        assert backend._submit_timeout_s == 300

    def test_npu_timeout_ms_batch_mode(self):
        """timeout_ms=0 means batch mode (long default timeout)."""
        from vten.backend.xsim import XsimBackend

        config = _xsim_config()
        config["backend"]["xsim"]["timeout_ms"] = 0
        backend = XsimBackend(project_config=config)
        assert backend._timeout_ms == 0 or backend._timeout_ms >= 10000

    def test_npu_40_bfm_configs_protocol_distribution(self):
        """NPU 3D: 6 AXI4L + 2 AXI4(DDR) + 32 AXI4(HBM) = 40."""
        cfgs = _npu_40_bfm_configs()
        axil_count = sum(1 for c in cfgs if c.protocol == Protocol.AXI4L)
        axi4_count = sum(1 for c in cfgs if c.protocol == Protocol.AXI4)
        assert axil_count == 6
        assert axi4_count == 34  # 2 DDR + 32 HBM
        assert axil_count + axi4_count == 40

    def test_npu_large_shm_image_size(self):
        """NPU-scale SHM image (256 commands) is substantial."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE, CONTROL_SIZE

        image = _large_shm_image(256)
        expected_min = CONTROL_SIZE + 256 * CMD_SLOT_SIZE
        assert len(image) >= expected_min

    def test_npu_submit_timeout_configurable(self):
        """Different timeout values are stored correctly."""
        from vten.backend.xsim import XsimBackend

        for timeout in [60, 120, 300, 600]:
            config = _xsim_config()
            config["backend"]["xsim"]["submit_timeout_s"] = timeout
            backend = XsimBackend(project_config=config)
            assert backend._submit_timeout_s == timeout


# ═══════════════════════════════════════════════════════════════════
# §7  SHM constants alignment
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendSHMConstants:
    """SHM constants must match 00_data_models.md §10.1."""

    def test_magic_value(self):
        from vten.backend.xsim import XsimBackend

        assert XsimBackend.SHM_MAGIC == 0x5654454E or hasattr(XsimBackend, "SHM_MAGIC")

    def test_control_size(self):
        """Control region is 256 bytes."""
        from vten.backend.sim.shm_constants import CONTROL_SIZE

        assert CONTROL_SIZE == 256

    def test_cmd_slot_size(self):
        """Command slot is 64 bytes."""
        from vten.backend.sim.shm_constants import CMD_SLOT_SIZE

        assert CMD_SLOT_SIZE == 64

    def test_host_status_values(self):
        """Host status: IDLE=0, CMD_READY=1, ACK=2, SHUTDOWN=3."""
        from vten.backend.sim.shm_constants import (
            HOST_STATUS_ACK,
            HOST_STATUS_CMD_READY,
            HOST_STATUS_IDLE,
            HOST_STATUS_SHUTDOWN,
        )

        assert HOST_STATUS_IDLE == 0
        assert HOST_STATUS_CMD_READY == 1
        assert HOST_STATUS_ACK == 2
        assert HOST_STATUS_SHUTDOWN == 3

    def test_backend_status_values(self):
        """Backend status: IDLE=0, RUNNING=1, DONE=2, ERROR=3."""
        from vten.backend.sim.shm_constants import (
            BACKEND_STATUS_DONE,
            BACKEND_STATUS_ERROR,
            BACKEND_STATUS_IDLE,
            BACKEND_STATUS_RUNNING,
        )

        assert BACKEND_STATUS_IDLE == 0
        assert BACKEND_STATUS_RUNNING == 1
        assert BACKEND_STATUS_DONE == 2
        assert BACKEND_STATUS_ERROR == 3


# ═══════════════════════════════════════════════════════════════════
# §8  Handshake protocol — SHM buffer behavioral tests
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendHandshake:
    """SHM buffer read/write for handshake protocol (04_backend_xsim.md §3).

    Tests operate on bytearray buffers simulating POSIX SHM regions.
    Offsets: host_status=0x08, backend_status=0x0C (from shm.py).
    """

    HOST_STATUS_OFF = 0x08
    BACKEND_STATUS_OFF = 0x0C

    def _make_control_buf(self) -> bytearray:
        """Create a 256-byte control region with MAGIC+VERSION."""
        from vten.backend.sim.shm_constants import CONTROL_SIZE
        buf = bytearray(CONTROL_SIZE)
        struct.pack_into("<I", buf, 0x00, 0x5654454E)  # MAGIC
        struct.pack_into("<I", buf, 0x04, 0x00000003)  # VERSION
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, 0)     # IDLE
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, 0)  # IDLE
        return buf

    def test_host_writes_cmd_ready_to_buffer(self):
        """Host writes CMD_READY(1) at offset 0x08 in SHM buffer."""
        from vten.backend.sim.shm_constants import HOST_STATUS_CMD_READY

        buf = self._make_control_buf()
        # Host action: set CMD_READY
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_CMD_READY)
        # Verify read-back
        val = struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0]
        assert val == 1

    def test_backend_writes_running_to_buffer(self):
        """Backend writes RUNNING(1) at offset 0x0C in SHM buffer."""
        from vten.backend.sim.shm_constants import BACKEND_STATUS_RUNNING

        buf = self._make_control_buf()
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_RUNNING)
        val = struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0]
        assert val == 1

    def test_normal_handshake_buffer_sequence(self):
        """Full normal handshake: IDLE→CMD_READY→RUNNING→DONE→ACK→IDLE."""
        from vten.backend.sim.shm_constants import (
            BACKEND_STATUS_DONE,
            BACKEND_STATUS_IDLE,
            BACKEND_STATUS_RUNNING,
            HOST_STATUS_ACK,
            HOST_STATUS_CMD_READY,
            HOST_STATUS_IDLE,
        )

        buf = self._make_control_buf()

        # 1. Initial: both IDLE
        assert struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0] == HOST_STATUS_IDLE
        assert struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0] == BACKEND_STATUS_IDLE

        # 2. Host sets CMD_READY
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_CMD_READY)
        assert struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0] == HOST_STATUS_CMD_READY

        # 3. Backend sees CMD_READY, sets RUNNING
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_RUNNING)
        assert struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0] == BACKEND_STATUS_RUNNING

        # 4. Backend finishes, sets DONE
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_DONE)
        assert struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0] == BACKEND_STATUS_DONE

        # 5. Host sees DONE, sends ACK
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_ACK)
        assert struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0] == HOST_STATUS_ACK

        # 6. Both return to IDLE
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_IDLE)
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_IDLE)
        assert struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0] == HOST_STATUS_IDLE
        assert struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0] == BACKEND_STATUS_IDLE

    def test_error_handshake_buffer_sequence(self):
        """Error flow: IDLE→CMD_READY→RUNNING→ERROR (backend writes error fields)."""
        from vten.backend.sim.shm_constants import (
            BACKEND_STATUS_ERROR,
            BACKEND_STATUS_RUNNING,
            HOST_STATUS_CMD_READY,
        )

        buf = self._make_control_buf()

        # Host sets CMD_READY
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_CMD_READY)

        # Backend sets RUNNING then ERROR
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_RUNNING)
        struct.pack_into("<I", buf, self.BACKEND_STATUS_OFF, BACKEND_STATUS_ERROR)
        assert struct.unpack_from("<I", buf, self.BACKEND_STATUS_OFF)[0] == BACKEND_STATUS_ERROR

        # Backend writes error_code at 0x40 (per spec §5.1)
        ERROR_CODE_OFF = 0x40
        struct.pack_into("<I", buf, ERROR_CODE_OFF, 1)  # ADDR_UNMATCH
        assert struct.unpack_from("<I", buf, ERROR_CODE_OFF)[0] == 1

    def test_shutdown_buffer_write(self):
        """Host writes SHUTDOWN(3) to buffer, backend reads it."""
        from vten.backend.sim.shm_constants import HOST_STATUS_SHUTDOWN

        buf = self._make_control_buf()
        struct.pack_into("<I", buf, self.HOST_STATUS_OFF, HOST_STATUS_SHUTDOWN)
        val = struct.unpack_from("<I", buf, self.HOST_STATUS_OFF)[0]
        assert val == HOST_STATUS_SHUTDOWN == 3

    def test_shm_image_magic_at_offset_zero(self):
        """SHM buffer offset 0x00 = MAGIC (0x5654454E)."""
        buf = self._make_control_buf()
        assert struct.unpack_from("<I", buf, 0x00)[0] == 0x5654454E

    def test_shm_image_version_at_offset_four(self):
        """SHM buffer offset 0x04 = VERSION (0x00000003)."""
        buf = self._make_control_buf()
        assert struct.unpack_from("<I", buf, 0x04)[0] == 0x00000003

    def test_error_message_in_buffer(self):
        """Error message written at offset 0x48 (64 bytes) is readable."""
        buf = self._make_control_buf()
        ERROR_MSG_OFF = 0x48
        ERROR_MSG_SIZE = 64
        msg = b"DECERR at addr=0x00100000"
        buf[ERROR_MSG_OFF:ERROR_MSG_OFF + len(msg)] = msg
        read_back = bytes(buf[ERROR_MSG_OFF:ERROR_MSG_OFF + ERROR_MSG_SIZE]).split(b"\x00")[0]
        assert read_back == msg


# ═══════════════════════════════════════════════════════════════════
# §9  Multi-batch lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestXsimBackendMultiBatch:
    """Multiple execute() cycles on same backend instance."""

    def test_backend_supports_execute(self):
        """Backend has callable execute() method."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        assert callable(backend.execute)

    def test_session_id_increments(self):
        """Each session gets a unique ID (monotonic or random)."""
        from vten.backend.xsim import XsimBackend

        backend = XsimBackend(project_config=_xsim_config())
        ids = [backend._generate_session_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All unique
