"""SimBackend — SHM handshake-based simulation backend base class.

Extracts common POSIX SHM + Named Semaphore protocol from XsimBackend.
Both xsim and verilator share this handshake; they differ only in
simulator process management.

Spec reference: 04_backend_xsim.md §1-6, 08_backend_abstraction.md §5.2, §7.2
"""

from __future__ import annotations

import abc
import ctypes
import ctypes.util
import os
import struct
import subprocess
import time
import uuid
from typing import TYPE_CHECKING

from vten.backend.base import Backend, BackendResult, CmdStats, raise_backend_error
from vten.errors import BackendError
from vten.errors import TimeoutError as VTenTimeoutError
from vten.runtime.shm import (
    BACKEND_STATUS_DONE,
    BACKEND_STATUS_ERROR,
    BACKEND_STATUS_IDLE,
    BUF_DESC_SIZE,
    HOST_STATUS_ACK,
    HOST_STATUS_CMD_READY,
    HOST_STATUS_IDLE,
    HOST_STATUS_SHUTDOWN,
    STATS_SLOT_SIZE,
)

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


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


# ── SimBackend ──


class SimBackend(Backend):
    """SHM handshake-based simulation backend base class.

    Implements POSIX SHM + Named Semaphore protocol (04_backend_xsim.md §3).
    Simulator-specific process management is delegated to subclasses.

    Class hierarchy (08_backend_abstraction.md §5.5):
        Backend (ABC)
        └── SimBackend (ABC) — SHM handshake common
            ├── XsimBackend — Vivado xsim process
            └── VerilatorBackend — Verilator binary
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

    def __init__(self, project_config: dict, backend_section: str) -> None:
        self._config = project_config
        cfg = project_config.get("backend", {}).get(backend_section, {})
        self._submit_timeout_s = cfg.get("submit_timeout_s", 300)
        self._timeout_ms = cfg.get("timeout_ms", 10000)

        # Runtime state — all None until submit
        self._process: subprocess.Popen | None = None
        self._shm = None  # multiprocessing.shared_memory.SharedMemory
        self._session_id: str | None = None
        self._sem_h2b: _PosixSemaphore | None = None
        self._sem_b2h: _PosixSemaphore | None = None

    # ── Backend ABC: execute() ──

    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Full lifecycle: submit SHM → start sim → wait → result."""
        try:
            self._submit_shm(compiled.shm_image, compiled.bfm_configs)
            return self._wait_completion()
        finally:
            self._shutdown_sim()

    # ── Optional fine-grained control ──

    def submit(self, compiled: CompiledResult) -> None:
        """Async submit: write SHM, launch sim, signal ready."""
        self._submit_shm(compiled.shm_image, compiled.bfm_configs)

    def wait(self) -> BackendResult:
        """Wait for completion and return result."""
        return self._wait_completion()

    def shutdown(self) -> None:
        """Send SHUTDOWN signal to backend, wait for process exit."""
        self._shutdown_sim()

    def cleanup(self) -> None:
        """Release all POSIX resources. Idempotent and exception-safe."""
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

    # ── Simulator process management (subclass override) ──

    @abc.abstractmethod
    def _start_simulator(self) -> None:
        """Launch the simulator subprocess. Sets self._process."""
        ...

    # ── SHM handshake internals ──

    def _generate_session_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _raise_backend_error(self, code: int, cmd_id: int, message: str) -> None:
        raise_backend_error(code, cmd_id, message)

    def _read_shm_u32(self, offset: int) -> int:
        """Read uint32 from SHM control region."""
        return struct.unpack_from("<I", self._shm.buf, offset)[0]

    def _write_shm_u32(self, offset: int, value: int) -> None:
        """Write uint32 to SHM control region."""
        struct.pack_into("<I", self._shm.buf, offset, value)

    def _submit_shm(self, shm_image: bytes, bfm_configs: list) -> None:
        """Write SHM image, create semaphores, launch sim, signal batch.

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

        # Start simulator subprocess
        self._start_simulator()

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

    def _wait_completion(self) -> BackendResult:
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

        # DONE: read stats, build buffer reader, send ACK
        stats = self._read_stats_from_shm()
        buffer_reader = self._make_buffer_reader()
        self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_ACK)

        return BackendResult(
            status=backend_status,
            stats=stats,
            _shm_reader=buffer_reader,
        )

    def _shutdown_sim(self) -> None:
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

        # Wait for process to exit
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

    def _make_buffer_reader(self):
        """Create a closure that reads buffer data from the live SHM segment.

        Parses buffer descriptors from the SHM to find the data_offset and size
        for a given buffer_id, then reads from the data region.

        The closure captures self._shm so it must be called before cleanup().
        """
        shm = self._shm
        if shm is None:
            return None

        buf = shm.buf
        num_buffers = struct.unpack_from("<I", buf, 0x14)[0]
        buf_desc_offset = struct.unpack_from("<Q", buf, 0x28)[0]
        data_region_offset = struct.unpack_from("<Q", buf, 0x30)[0]

        # Pre-parse all buffer descriptors: buffer_id → (data_offset, size)
        desc_map: dict[int, tuple[int, int]] = {}
        for i in range(num_buffers):
            base = buf_desc_offset + i * BUF_DESC_SIZE
            bid = struct.unpack_from("<H", buf, base + 0x00)[0]
            size = struct.unpack_from("<I", buf, base + 0x04)[0]
            data_offset = struct.unpack_from("<Q", buf, base + 0x08)[0]
            desc_map[bid] = (data_offset, size)

        def _read(buffer_id: int) -> bytes:
            if buffer_id not in desc_map:
                return b""
            data_offset, size = desc_map[buffer_id]
            start = data_region_offset + data_offset
            end = start + size
            if end > len(shm.buf):
                return b""
            return bytes(shm.buf[start:end])

        return _read

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
