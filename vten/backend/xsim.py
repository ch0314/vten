"""XsimBackend: Vivado xsim backend adapter.

Implements POSIX SHM + Named Semaphore handshake protocol.
Spec reference: 04_backend_xsim.md §1-6, 06_codegen_and_cli.md §4.4
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import struct
import subprocess
import time
import uuid

from vten.backend.base import Backend, BackendResult, CmdStats, raise_backend_error
from vten.errors import BackendError
from vten.errors import TimeoutError as VTenTimeoutError
from vten.runtime.shm import (
    BACKEND_STATUS_DONE,
    BACKEND_STATUS_ERROR,
    BACKEND_STATUS_IDLE,
    HOST_STATUS_ACK,
    HOST_STATUS_CMD_READY,
    HOST_STATUS_IDLE,
    HOST_STATUS_SHUTDOWN,
    STATS_SLOT_SIZE,
)


# ── POSIX Named Semaphore wrapper ──


class _PosixSemaphore:
    """Thin ctypes wrapper around POSIX named semaphores.

    Falls back gracefully when POSIX APIs are unavailable (non-Linux, etc.).
    """

    _lib = None
    _available = None

    @classmethod
    def _load(cls) -> bool:
        if cls._available is not None:
            return cls._available
        for libname in ("pthread", "rt", "c"):
            path = ctypes.util.find_library(libname)
            if path:
                try:
                    cls._lib = ctypes.CDLL(path, use_errno=True)
                    if hasattr(cls._lib, "sem_open"):
                        cls._available = True
                        return True
                except OSError:
                    continue
        cls._available = False
        return False

    def __init__(self, name: str, *, create: bool = False) -> None:
        self._name = name.encode("utf-8")
        self._sem = None

        if not self._load():
            return

        lib = self._lib
        lib.sem_open.restype = ctypes.c_void_p
        if create:
            self._sem = lib.sem_open(
                self._name, ctypes.c_int(os.O_CREAT), ctypes.c_uint(0o644), ctypes.c_uint(0),
            )
        else:
            self._sem = lib.sem_open(self._name, ctypes.c_int(0))

        SEM_FAILED = ctypes.c_void_p(-1).value
        if self._sem is None or self._sem == SEM_FAILED:
            self._sem = None

    def post(self) -> None:
        if self._sem is None:
            return
        self._lib.sem_post(ctypes.c_void_p(self._sem))

    def wait(self) -> bool:
        if self._sem is None:
            return True
        return self._lib.sem_wait(ctypes.c_void_p(self._sem)) == 0

    def timedwait(self, timeout_s: float) -> bool:
        """Wait with timeout. Returns True if acquired, False on timeout."""
        if self._sem is None:
            return True

        if not hasattr(self._lib, "sem_timedwait"):
            return self.wait()

        class _timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        deadline = time.time() + timeout_s
        ts = _timespec()
        ts.tv_sec = int(deadline)
        ts.tv_nsec = int((deadline - int(deadline)) * 1_000_000_000)

        result = self._lib.sem_timedwait(ctypes.c_void_p(self._sem), ctypes.byref(ts))
        return result == 0

    def close(self) -> None:
        if self._sem is None:
            return
        try:
            self._lib.sem_close(ctypes.c_void_p(self._sem))
        except Exception:
            pass
        self._sem = None

    def unlink(self) -> None:
        try:
            self._lib.sem_unlink(self._name)
        except Exception:
            pass


# ── XsimBackend ──


class XsimBackend(Backend):
    """Backend adapter for Vivado xsim simulator.

    Manages POSIX SHM segment and named semaphore pair for host↔backend
    communication. Launches xsim as a subprocess.

    Handshake protocol (04_backend_xsim.md §3):
    1. Host: shm_open, write image, sem_open, start xsim
    2. Backend (DPI-C): vten_shm_init → sem_post(b2h) "ready"
    3. Host: sem_wait(b2h), write CMD_READY, sem_post(h2b)
    4. Backend: execute → sem_post(b2h) "done/error"
    5. Host: sem_wait(b2h), read result, send ACK
    6. Shutdown: host_status=SHUTDOWN, sem_post(h2b)
    """

    SHM_MAGIC = 0x5654454E  # "VTEN"

    # Control region offsets (00_data_models.md §10.2)
    HOST_STATUS_OFFSET = 0x08
    BACKEND_STATUS_OFFSET = 0x0C
    ERROR_CODE_OFFSET = 0x40
    ERROR_CMD_ID_OFFSET = 0x44
    ERROR_MSG_OFFSET = 0x48
    ERROR_MSG_SIZE = 64
    STATS_ENABLED_OFFSET = 0x88

    def __init__(self, project_config: dict) -> None:
        self._config = project_config
        xsim_cfg = project_config.get("backend", {}).get("xsim", {})
        self._vivado_path = xsim_cfg.get("vivado_path", "")
        self._submit_timeout_s = xsim_cfg.get("submit_timeout_s", 300)
        self._timeout_ms = xsim_cfg.get("timeout_ms", 10000)

        # Runtime state — all None until submit()
        self._process: subprocess.Popen | None = None
        self._shm = None  # multiprocessing.shared_memory.SharedMemory
        self._session_id: str | None = None
        self._sem_h2b: _PosixSemaphore | None = None
        self._sem_b2h: _PosixSemaphore | None = None

    def __enter__(self) -> XsimBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        self.cleanup()

    def _generate_session_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _raise_backend_error(self, code: int, cmd_id: int, message: str) -> None:
        raise_backend_error(code, cmd_id, message)

    def _start_xsim(self) -> None:
        """Launch xsim subprocess with session_id plusarg.

        Note: --sv_lib is an xelab option, not xsim.
        The DPI-C library is linked during elaboration.
        xsim must run from the directory containing xsim.dir/.
        """
        vivado_path = self._vivado_path
        if vivado_path:
            xsim_bin = os.path.join(vivado_path, "bin", "xsim")
        else:
            xsim_bin = "xsim"

        rtl_cfg = self._config.get("rtl", {})
        top_module = rtl_cfg.get("tb_module", "tb_top")

        # xsim must run from directory containing xsim.dir/
        # Priority: config key > [backend.xsim].xsim_dir > _project_dir > cwd
        xsim_cfg = self._config.get("backend", {}).get("xsim", {})
        xsim_cwd = self._config.get("_xsim_dir",
                       xsim_cfg.get("xsim_dir",
                           self._config.get("_project_dir", ".")))
        # Resolve relative paths against project dir
        project_dir = self._config.get("_project_dir", ".")
        if not os.path.isabs(xsim_cwd):
            xsim_cwd = os.path.normpath(os.path.join(project_dir, xsim_cwd))

        cmd = [
            xsim_bin, top_module,
            "--runall",
            "--testplusarg", f"SESSION_ID={self._session_id}",
            "--testplusarg", f"TIMEOUT_MS={self._timeout_ms}",
        ]

        self._process = subprocess.Popen(
            cmd,
            cwd=xsim_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _read_shm_u32(self, offset: int) -> int:
        """Read uint32 from SHM control region."""
        return struct.unpack_from("<I", self._shm.buf, offset)[0]

    def _write_shm_u32(self, offset: int, value: int) -> None:
        """Write uint32 to SHM control region."""
        struct.pack_into("<I", self._shm.buf, offset, value)

    def _read_stats_from_shm(self) -> list[CmdStats]:
        """Parse per-command stats from SHM Stats Region."""
        stats: list[CmdStats] = []
        if self._shm is None:
            return stats

        num_commands = self._read_shm_u32(0x10)
        stats_offset = struct.unpack_from("<Q", self._shm.buf, 0x20)[0]

        for i in range(num_commands):
            base = stats_offset + i * STATS_SLOT_SIZE
            if base + STATS_SLOT_SIZE > len(self._shm.buf):
                break
            status = struct.unpack_from("<B", self._shm.buf, base)[0]
            vals = struct.unpack_from("<7i", self._shm.buf, base + 4)
            stats.append(CmdStats(
                cmd_id=i,
                status=status,
                issue_cycle=vals[0],
                commit_cycle=vals[1],
                first_active_cycle=vals[2],
                last_active_cycle=vals[3],
                active_cycles=vals[4],
                total_beats=vals[5],
                stall_cycles=vals[6],
            ))
        return stats

    # ── Backend ABC implementation ──

    def submit(self, shm_image: bytes, bfm_configs: list) -> None:
        """Write SHM image, create semaphores, launch xsim, signal batch.

        Implements handshake steps [1]-[4] from 04_backend_xsim.md §3.
        """
        from multiprocessing.shared_memory import SharedMemory

        self._session_id = self._generate_session_id()
        shm_name = f"vten_{self._session_id}"

        # Step [1]: Create POSIX SHM and write image
        if shm_image is not None:
            self._shm = SharedMemory(name=shm_name, create=True, size=len(shm_image))
            self._shm.buf[:len(shm_image)] = shm_image
            # Ensure host_status = IDLE
            self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_IDLE)

        # Create named semaphore pair
        self._sem_h2b = _PosixSemaphore(f"/vten_{self._session_id}_h2b", create=True)
        self._sem_b2h = _PosixSemaphore(f"/vten_{self._session_id}_b2h", create=True)

        # Start xsim subprocess
        self._start_xsim()

        # Step [3]: Wait for backend ready signal (b2h)
        if not self._sem_b2h.timedwait(30.0):
            raise VTenTimeoutError("backend did not signal ready within 30s")

        # Verify backend_status == IDLE after init
        if self._shm is not None:
            backend_status = self._read_shm_u32(self.BACKEND_STATUS_OFFSET)
            if backend_status != BACKEND_STATUS_IDLE:
                self._raise_backend_error(
                    backend_status, 0, f"unexpected backend status {backend_status} after init",
                )

        # Step [4]: Signal CMD_READY to backend
        if self._shm is not None:
            self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_CMD_READY)
        self._sem_h2b.post()

    def wait(self) -> BackendResult:
        """Wait for backend completion, read results.

        Implements handshake steps [8]-[9] from 04_backend_xsim.md §3.
        """
        # No SHM / semaphore → stub mode (for mocked tests)
        if self._sem_b2h is None or self._shm is None:
            return BackendResult(status=BACKEND_STATUS_DONE)

        # Step [8]: Wait for backend done/error signal
        if not self._sem_b2h.timedwait(self._submit_timeout_s):
            raise VTenTimeoutError(
                f"backend did not complete within {self._submit_timeout_s}s",
            )

        backend_status = self._read_shm_u32(self.BACKEND_STATUS_OFFSET)

        # Handle ERROR
        if backend_status == BACKEND_STATUS_ERROR:
            error_code = self._read_shm_u32(self.ERROR_CODE_OFFSET)
            error_cmd_id = self._read_shm_u32(self.ERROR_CMD_ID_OFFSET)
            error_msg_raw = bytes(
                self._shm.buf[self.ERROR_MSG_OFFSET:self.ERROR_MSG_OFFSET + self.ERROR_MSG_SIZE]
            )
            error_msg = error_msg_raw.split(b"\x00")[0].decode("utf-8", errors="replace")

            # Still send ACK before raising
            self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_ACK)
            self._raise_backend_error(error_code, error_cmd_id, error_msg)

        # DONE: read stats, send ACK
        stats = self._read_stats_from_shm()
        self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_ACK)

        return BackendResult(
            status=backend_status,
            stats=stats,
        )

    def shutdown(self) -> None:
        """Send SHUTDOWN signal to backend, wait for process exit.

        Implements handshake step [9] from 04_backend_xsim.md §3.
        """
        # Signal SHUTDOWN via SHM + semaphore
        if self._shm is not None:
            try:
                self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_SHUTDOWN)
            except Exception:
                pass
        if self._sem_h2b is not None:
            self._sem_h2b.post()

        # Wait for xsim process to exit
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

    def cleanup(self) -> None:
        """Release all POSIX resources. Idempotent and exception-safe.

        Implements handshake step [11] from 04_backend_xsim.md §3.
        """
        # Close and unlink semaphores
        if self._sem_h2b is not None:
            try:
                self._sem_h2b.close()
                self._sem_h2b.unlink()
            except Exception:
                pass
            self._sem_h2b = None

        if self._sem_b2h is not None:
            try:
                self._sem_b2h.close()
                self._sem_b2h.unlink()
            except Exception:
                pass
            self._sem_b2h = None

        # Close and unlink SHM
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        # Terminate process if still alive
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None

        self._session_id = None
