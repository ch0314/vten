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
import logging
import os
import struct
import subprocess
import time
import uuid
from typing import TYPE_CHECKING

from vten.backend.base import Backend, BackendResult, CmdStats, raise_backend_error
from vten.errors import BackendError
from vten.errors import TimeoutError as VTenTimeoutError

logger = logging.getLogger(__name__)
from vten.runtime.shm import (
    BACKEND_STATUS_DONE,
    BACKEND_STATUS_ERROR,
    BACKEND_STATUS_IDLE,
    BACKEND_STATUS_RUNNING,
    BUF_DESC_SIZE,
    CMD_SLOT_SIZE,
    HOST_STATUS_ACK,
    HOST_STATUS_CMD_READY,
    HOST_STATUS_IDLE,
    HOST_STATUS_SHUTDOWN,
    STATS_SLOT_SIZE,
)

# Diagnostic name maps derived from enums (single source of truth)
from vten.spec.models import CommandStatus, OpCode

_STATUS_NAMES = {e.value: e.name for e in CommandStatus}
_OPCODE_NAMES = {e.value: e.name for e in OpCode}
_BACKEND_STATUS_NAMES = {
    BACKEND_STATUS_IDLE: "IDLE",
    BACKEND_STATUS_RUNNING: "RUNNING",
    BACKEND_STATUS_DONE: "DONE",
    BACKEND_STATUS_ERROR: "ERROR",
}

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
            self._probe_buffer_map = getattr(compiled, "probe_buffer_map", {})
            self._submit_shm(compiled.shm_image, compiled.bfm_configs)
            return self._wait_completion()
        finally:
            self._shutdown_sim()
            self._release_posix_resources()

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

    def _release_posix_resources(self) -> None:
        """Release SHM and semaphores. Called after each execute() cycle."""
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

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

        self._session_id = None

    def cleanup(self) -> None:
        """Release all POSIX resources. Idempotent and exception-safe."""
        self._release_posix_resources()

        # Terminate process if still alive
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None

    # ── Simulator process management (subclass override) ──

    @abc.abstractmethod
    def _start_simulator(self) -> None:
        """Launch the simulator subprocess. Sets self._process."""
        ...

    # ── SHM handshake internals ──

    def _generate_session_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def _raise_backend_error(self, code: int, cmd_id: int, message: str) -> None:
        # Enrich ProbeMismatchError with mismatch details from JSONL file
        if code == 8:  # ERR_PROBE_MISMATCH
            mismatches = self._parse_mismatch_file()
            if mismatches:
                beat_index = mismatches[0].get("beat", 0)
                from vten.errors import ProbeMismatchError
                raise ProbeMismatchError(
                    f"{message} (cmd_id={cmd_id})",
                    cmd_id=cmd_id,
                    beat_index=beat_index,
                    mismatches=mismatches,
                    context={"error_code": code, "cmd_id": cmd_id},
                )
        raise_backend_error(code, cmd_id, message)

    def _parse_mismatch_file(self) -> list[dict]:
        """Parse mismatches.jsonl written by the C bridge."""
        import json
        mismatch_dir = self._config.get("_mismatch_dir")
        if not mismatch_dir:
            return []
        mismatch_path = os.path.join(str(mismatch_dir), "mismatches.jsonl")
        if not os.path.isfile(mismatch_path):
            return []
        results = []
        try:
            with open(mismatch_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("failed to parse mismatches.jsonl: %s", e)
        return results

    def _read_shm_u32(self, offset: int) -> int:
        """Read uint32 from SHM control region."""
        return struct.unpack_from("<I", self._shm.buf, offset)[0]

    def _write_shm_u32(self, offset: int, value: int) -> None:
        """Write uint32 to SHM control region."""
        struct.pack_into("<I", self._shm.buf, offset, value)

    def _resize_shm(self, new_size: int) -> None:
        """Grow POSIX SHM via ftruncate and re-mmap on Python side.

        The C bridge will detect the change via fstat in vten_shm_remap(),
        called at S_LOAD_BATCH entry.
        """
        from multiprocessing.shared_memory import SharedMemory

        old_size = self._shm.size
        shm_name = self._shm.name  # e.g. "vten_abc123"

        logger.info("[session] resizing SHM: %d → %d bytes", old_size, new_size)

        # ftruncate the underlying POSIX SHM object to grow it.
        # SharedMemory doesn't expose fd, so open it directly.
        fd = os.open(f"/dev/shm/{shm_name}", os.O_RDWR)
        try:
            os.ftruncate(fd, new_size)
        finally:
            os.close(fd)

        # Close old Python SharedMemory (releases old mmap, keeps SHM object)
        self._shm.close()

        # Re-open at new size (attach to existing, don't create)
        self._shm = SharedMemory(name=shm_name, create=False)

    def _submit_shm(self, shm_image: bytes, bfm_configs: list) -> None:
        """Write SHM image, create semaphores, launch sim, signal batch.

        Implements handshake steps [1]-[4] from 04_backend_xsim.md §3.
        """
        from multiprocessing.shared_memory import SharedMemory

        self._session_id = self._generate_session_id()
        shm_name = f"vten_{self._session_id}"

        # Step [1]: Create POSIX SHM and write image
        if shm_image is not None:
            logger.log(5, "[handshake 1] creating SHM: name=%s, size=%d bytes",
                         shm_name, len(shm_image))
            self._shm = SharedMemory(name=shm_name, create=True, size=len(shm_image))
            self._shm.buf[:len(shm_image)] = shm_image
            # Ensure host_status = IDLE
            self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_IDLE)

        # Create named semaphore pair
        self._sem_h2b = _PosixSemaphore(f"/vten_{self._session_id}_h2b", create=True)
        self._sem_b2h = _PosixSemaphore(f"/vten_{self._session_id}_b2h", create=True)
        logger.log(5, "[handshake 1] semaphores created: h2b, b2h (session=%s)",
                     self._session_id)

        # Start simulator subprocess
        logger.log(5, "[handshake 2] starting simulator")
        self._start_simulator()
        logger.log(5, "[handshake 2] simulator process started (pid=%s)",
                     self._process.pid if self._process else "?")

        # Step [3]: Wait for backend ready signal (b2h)
        init_timeout = min(self._submit_timeout_s, 120)
        logger.log(5, "[handshake 3] waiting for backend ready (timeout=%ds)", init_timeout)
        if not self._sem_b2h.timedwait(init_timeout):
            # Check if simulator crashed before signaling
            proc_status = self._describe_process_state()
            sim_output = self._drain_simulator_output()
            msg = (f"backend did not signal ready within {init_timeout}s"
                   f" (session={self._session_id}, {proc_status})")
            logger.error("%s", msg)
            if sim_output:
                logger.error("simulator output tail:\n%s", sim_output[-2000:])
            raise VTenTimeoutError(msg)

        logger.log(5, "backend ready (session=%s)", self._session_id)

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
        logger.log(5, "[handshake 4] CMD_READY signaled")

    # Progress polling interval (seconds)
    _PROGRESS_POLL_INTERVAL = 2.0

    def _wait_completion(self) -> BackendResult:
        """Wait for backend completion with progress monitoring.

        Polls SHM stats region periodically to report command progress,
        then reads full results on completion.

        In GUI mode, stays alive across error/done/restart cycles so the
        user can interactively restart and re-run in xsim.  Only exits
        when the xsim process terminates (user closes GUI) or Ctrl-C.

        Implements handshake steps [8]-[9] from 04_backend_xsim.md §3.
        """
        # No SHM / semaphore → stub mode (for mocked tests)
        if self._sem_b2h is None or self._shm is None:
            return BackendResult(status=BACKEND_STATUS_DONE)

        gui_mode = self._config.get("_gui")

        # Step [8]: Wait for backend done/error signal with progress polling
        # GUI mode: extend timeout to 24h — user controls simulation interactively
        effective_timeout = self._submit_timeout_s
        if gui_mode:
            effective_timeout = 86400  # 24 hours

        # GUI state: accumulate across restart cycles
        gui_last_error: tuple | None = None   # (code, cmd_id, msg)
        gui_last_result: BackendResult | None = None

        while True:  # GUI restart loop (single iteration in normal mode)
            logger.log(5, "[handshake 8] waiting for backend completion (timeout=%ds)",
                         effective_timeout)
            deadline = time.monotonic() + effective_timeout
            last_progress_str = ""
            poll_count = 0
            process_exited = False

            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break

                    wait_time = min(self._PROGRESS_POLL_INTERVAL, remaining)
                    if self._sem_b2h.timedwait(wait_time):
                        # Semaphore acquired — backend signaled completion
                        break
                    else:
                        # Timeout on this poll — read progress from SHM
                        poll_count += 1
                        progress_str = self._format_progress()
                        if progress_str and progress_str != last_progress_str:
                            elapsed = time.monotonic() - (deadline - effective_timeout)
                            logger.info("  [%.1fs] %s", elapsed, progress_str)
                            last_progress_str = progress_str
                        # Check if simulator process died
                        if self._process is not None and self._process.poll() is not None:
                            process_exited = True
                            break
                else:
                    # Outer while exhausted remaining time → timeout
                    self._handle_timeout()
            except KeyboardInterrupt:
                logger.info("Ctrl-C detected, shutting down simulator...")
                self._shutdown_sim()
                raise

            backend_status = self._read_shm_u32(self.BACKEND_STATUS_OFFSET)
            logger.log(5, "[handshake 8] backend status=%d", backend_status)
            process_alive = self._process is None or self._process.poll() is None

            # Handle ERROR
            if backend_status == BACKEND_STATUS_ERROR:
                error_code = self._read_shm_u32(self.ERROR_CODE_OFFSET)
                error_cmd_id = self._read_shm_u32(self.ERROR_CMD_ID_OFFSET)
                error_msg_raw = bytes(
                    self._shm.buf[self.ERROR_MSG_OFFSET:self.ERROR_MSG_OFFSET + self.ERROR_MSG_SIZE]
                )
                error_msg = error_msg_raw.split(b"\x00")[0].decode("utf-8", errors="replace")
                logger.error("backend error: code=%d, cmd_id=%d, msg=%s",
                             error_code, error_cmd_id, error_msg)

                if gui_mode and process_alive:
                    # GUI: log error, reset SHM for restart, keep session alive
                    gui_last_error = (error_code, error_cmd_id, error_msg)
                    logger.info("[gui] error logged — restart xsim or close to finish")
                    self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_CMD_READY)
                    self._sem_h2b.post()
                    continue  # Wait for restart or process exit

                # Normal mode or process exited: raise
                self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_ACK)
                self._raise_backend_error(error_code, error_cmd_id, error_msg)

            # IDLE after b2h signal → xsim restarted (re-called vten_shm_init)
            if backend_status == BACKEND_STATUS_IDLE:
                if gui_mode and process_alive:
                    logger.info("[gui] simulator restarted — re-sending CMD_READY")
                    self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_CMD_READY)
                    self._sem_h2b.post()
                    continue

            # RUNNING: either restart (b2h from new vten_shm_init, old status)
            # or process exited while sim was stuck / at $stop
            if backend_status == BACKEND_STATUS_RUNNING:
                if gui_mode and process_alive and not process_exited:
                    # b2h signal received but status still RUNNING →
                    # xsim restarted (vten_shm_init posted b2h, old RUNNING
                    # status wasn't cleared). Reset and re-send CMD_READY.
                    logger.info("[gui] simulator restarted — re-sending CMD_READY")
                    self._write_shm_u32(self.BACKEND_STATUS_OFFSET,
                                        BACKEND_STATUS_IDLE)
                    self._write_shm_u32(self.HOST_STATUS_OFFSET,
                                        HOST_STATUS_CMD_READY)
                    self._sem_h2b.post()
                    continue
                if not process_alive:
                    if gui_mode:
                        # User closed xsim GUI (possibly at $stop).
                        # Return partial stats — runner will report FAIL
                        # if verifications didn't pass.
                        logger.info("[gui] xsim closed by user (status=RUNNING)")
                        if gui_last_error:
                            code, cmd_id, msg = gui_last_error
                            self._write_shm_u32(self.HOST_STATUS_OFFSET,
                                                HOST_STATUS_ACK)
                            self._raise_backend_error(code, cmd_id, msg)
                        stats = self._read_stats_from_shm()
                        return BackendResult(
                            status=BACKEND_STATUS_DONE,
                            stats=stats,
                        )
                    self._handle_timeout()
                self._handle_timeout()

            # DONE: read stats, build buffer reader
            stats = self._read_stats_from_shm()
            buffer_reader = self._make_buffer_reader()

            if gui_mode and process_alive:
                # GUI: save result, reset SHM for restart, keep session alive
                gui_last_result = BackendResult(
                    status=backend_status,
                    stats=stats,
                    _shm_reader=buffer_reader,
                )
                logger.info("[gui] simulation done — restart xsim or close to finish")
                self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_CMD_READY)
                self._sem_h2b.post()
                continue

            # Normal mode or process exited: return result
            self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_ACK)
            logger.log(5, "backend completed: %d command stats read", len(stats))

            return BackendResult(
                status=backend_status,
                stats=stats,
                _shm_reader=buffer_reader,
            )

        # GUI loop exited (should not normally reach here)
        if gui_last_error:
            code, cmd_id, msg = gui_last_error
            self._raise_backend_error(code, cmd_id, msg)
        if gui_last_result:
            return gui_last_result
        return BackendResult(status=BACKEND_STATUS_DONE)

    # ── Progress monitoring ──

    def _read_command_metadata(self) -> list[dict]:
        """Read opcode and interface_id for each command from SHM Command Region."""
        if self._shm is None:
            return []
        buf = self._shm.buf
        num_commands = self._read_shm_u32(0x10)
        cmd_offset = struct.unpack_from("<Q", buf, 0x18)[0]

        cmds = []
        for i in range(min(num_commands, 256)):
            base = cmd_offset + i * CMD_SLOT_SIZE
            if base + CMD_SLOT_SIZE > len(buf):
                break
            opcode = struct.unpack_from("<H", buf, base + 0x00)[0]
            iface_id = struct.unpack_from("<H", buf, base + 0x04)[0]
            reg_offset = struct.unpack_from("<I", buf, base + 0x18)[0]
            reg_mask = struct.unpack_from("<I", buf, base + 0x20)[0]
            reg_expected = struct.unpack_from("<I", buf, base + 0x24)[0]
            cmds.append({
                "cmd_id": i,
                "opcode": opcode,
                "opcode_name": _OPCODE_NAMES.get(opcode, f"?{opcode}"),
                "interface_id": iface_id,
                "reg_offset": reg_offset,
                "reg_mask": reg_mask,
                "reg_expected": reg_expected,
            })
        return cmds

    def _read_diagnostic_snapshot(self) -> dict:
        """Read full diagnostic state from SHM for timeout analysis."""
        if self._shm is None:
            return {}

        backend_status = self._read_shm_u32(self.BACKEND_STATUS_OFFSET)
        error_code = self._read_shm_u32(self.ERROR_CODE_OFFSET)
        num_commands = self._read_shm_u32(0x10)

        stats = self._read_stats_from_shm()
        cmd_meta = self._read_command_metadata()

        # Merge stats + metadata
        commands = []
        for i in range(len(stats)):
            s = stats[i]
            meta = cmd_meta[i] if i < len(cmd_meta) else {}
            status_name = _STATUS_NAMES.get(s.status, f"?{s.status}")
            commands.append({
                "cmd_id": s.cmd_id,
                "opcode": meta.get("opcode_name", "?"),
                "interface_id": meta.get("interface_id", -1),
                "status": status_name,
                "status_code": s.status,
                "issue_cycle": s.issue_cycle,
                "commit_cycle": s.commit_cycle,
                "last_active_cycle": s.last_active_cycle,
                "active_cycles": s.active_cycles,
                "total_beats": s.total_beats,
                "stall_cycles": s.stall_cycles,
                "reg_offset": meta.get("reg_offset", 0),
                "reg_mask": meta.get("reg_mask", 0),
                "reg_expected": meta.get("reg_expected", 0),
            })

        return {
            "backend_status": _BACKEND_STATUS_NAMES.get(backend_status, f"?{backend_status}"),
            "error_code": error_code,
            "num_commands": num_commands,
            "commands": commands,
        }

    def _format_progress(self) -> str:
        """Format a one-line progress string from current SHM stats."""
        if self._shm is None:
            return ""
        try:
            stats = self._read_stats_from_shm()
        except Exception:
            return ""

        if not stats:
            return ""

        total = len(stats)
        committed = sum(1 for s in stats if s.status >= 3)  # COMMITTED or ERROR
        issued = sum(1 for s in stats if s.status == 1)  # ISSUED (in-flight)
        pending = sum(1 for s in stats if s.status == 0)  # PENDING

        if committed == total:
            return f"{committed}/{total} commands done"

        # Find the stuck command (ISSUED with highest stall)
        stuck_info = ""
        if issued > 0:
            cmd_meta = self._read_command_metadata()
            for s in stats:
                if s.status == 1:  # ISSUED
                    meta = cmd_meta[s.cmd_id] if s.cmd_id < len(cmd_meta) else {}
                    op_name = meta.get("opcode_name", "?")
                    stuck_info = f", {op_name} cmd#{s.cmd_id}"
                    if s.stall_cycles > 100:
                        stuck_info += f" stalled {s.stall_cycles} cyc"
                    break

        return f"{committed}/{total} commands done ({issued} active, {pending} waiting{stuck_info})"

    def _format_timeout_report(self, diag: dict, elapsed: float) -> str:
        """Format a structured timeout diagnostic report."""
        lines = [
            f"Timeout after {elapsed:.1f}s (session={self._session_id})",
            f"  Backend status: {diag.get('backend_status', '?')}",
        ]

        commands = diag.get("commands", [])
        if not commands:
            lines.append("  No command data available")
            return "\n".join(lines)

        total = len(commands)
        committed = sum(1 for c in commands if c["status_code"] >= 3)
        issued = [c for c in commands if c["status_code"] == 1]
        pending = [c for c in commands if c["status_code"] == 0]

        lines.append(f"  Commands: {total} total, {committed} committed, "
                     f"{len(issued)} issued (stuck), {len(pending)} pending")

        # Command table
        lines.append("")
        lines.append("  ID  Op          Interface  Status     Cycles       Note")
        lines.append("  --- ----------- --------- ---------- ------------ ----")
        for c in commands:
            iface = str(c["interface_id"]) if c["interface_id"] >= 0 else "-"
            status = c["status"]
            cycles = ""
            note = ""

            if c["status_code"] >= 3:  # COMMITTED
                if c["issue_cycle"] or c["commit_cycle"]:
                    cycles = f"{c['issue_cycle']}-{c['commit_cycle']}"
                if c["total_beats"]:
                    note = f"{c['total_beats']} beats"
            elif c["status_code"] == 1:  # ISSUED (stuck)
                cycles = f"{c['issue_cycle']}-..."
                note = "STUCK"
                if c["stall_cycles"] > 0:
                    note += f" ({c['stall_cycles']} stall cyc)"
            elif c["status_code"] == 0:  # PENDING
                cycles = "-"
                # Find dependencies (not available from SHM, just note)
                note = "waiting"

            lines.append(f"  {c['cmd_id']:>3d}  {c['opcode']:<11s} {iface:>9s}  "
                         f"{status:<10s} {cycles:<12s} {note}")

        # Stuck command detail
        for c in issued:
            lines.append("")
            op = c["opcode"]
            detail = f"  >> Stuck: cmd#{c['cmd_id']} {op}"
            if op == "POLL_REG":
                detail += (f" (addr=0x{c['reg_offset']:04x}, "
                           f"mask=0x{c['reg_mask']:08x}, "
                           f"expected=0x{c['reg_expected']:08x})")
            detail += f"\n     Issued at cycle {c['issue_cycle']}"
            if c["last_active_cycle"]:
                detail += f", last active at cycle {c['last_active_cycle']}"
            if c["stall_cycles"]:
                detail += f", stalled for {c['stall_cycles']} cycles"
            lines.append(detail)

        return "\n".join(lines)

    def _handle_timeout(self) -> None:
        """Read diagnostic snapshot from SHM and raise structured TimeoutError."""
        elapsed = self._submit_timeout_s
        proc_status = self._describe_process_state()
        sim_output = self._drain_simulator_output()

        # Read diagnostic snapshot BEFORE destroying anything
        diag = {}
        try:
            diag = self._read_diagnostic_snapshot()
        except Exception:
            pass

        # Format structured report
        if diag and diag.get("commands"):
            report = self._format_timeout_report(diag, elapsed)
            logger.error("%s", report)
        else:
            logger.error("backend did not complete within %ds (session=%s, %s)",
                         elapsed, self._session_id, proc_status)

        if sim_output:
            logger.error("simulator output tail:\n%s", sim_output[-2000:])

        msg = (f"backend did not complete within {elapsed}s"
               f" (session={self._session_id}, {proc_status})")
        raise VTenTimeoutError(msg, context={"diagnosis": diag})

    def _shutdown_sim(self) -> None:
        """Send SHUTDOWN signal to backend, wait for process exit.

        Implements handshake step [9] from 04_backend_xsim.md §3.
        """
        logger.log(5, "[handshake 9] sending SHUTDOWN signal")
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
                logger.log(5, "simulator exited (rc=%d)", self._process.returncode)
            except subprocess.TimeoutExpired:
                logger.warning("simulator did not exit in 10s, sending SIGTERM")
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("simulator did not exit after SIGTERM, sending SIGKILL")
                    self._process.kill()
                    self._process.wait()

            # Drain simulator output after exit
            sim_output = self._drain_simulator_output()
            if self._process.returncode and self._process.returncode != 0:
                logger.warning("simulator exited with code %d", self._process.returncode)
                if sim_output:
                    logger.log(5, "simulator output:\n%s", sim_output[-2000:])

    def _describe_process_state(self) -> str:
        """Describe the simulator process state for diagnostics."""
        if self._process is None:
            return "process=None"
        rc = self._process.poll()
        if rc is None:
            return f"process=running (pid={self._process.pid})"
        return f"process=exited (rc={rc})"

    def _drain_simulator_output(self) -> str:
        """Read remaining stdout/stderr from simulator process.

        Subclasses may override for simulator-specific handling.
        Returns combined output string for logging.
        """
        if self._process is None:
            return ""
        parts: list[str] = []
        try:
            if self._process.stdout:
                stdout = self._process.stdout.read()
                if stdout:
                    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
                    parts.append(f"[stdout]\n{text}")
            if self._process.stderr:
                stderr = self._process.stderr.read()
                if stderr:
                    text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
                    parts.append(f"[stderr]\n{text}")
        except Exception:
            pass
        return "\n".join(parts)

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
            if end <= len(buf):
                return bytes(buf[start:end])
            return b""

        return _read

    # ── Session protocol (multi-batch) ──

    @property
    def supports_session(self) -> bool:
        return True

    def open_session(self, compiled: CompiledResult) -> None:
        """Open persistent session: create SHM, start sim, submit first batch."""
        self._probe_buffer_map = getattr(compiled, "probe_buffer_map", {})
        self._submit_shm(compiled.shm_image, compiled.bfm_configs)
        self._session_active = True

    def submit_batch(self, compiled: CompiledResult) -> None:
        """Submit a new batch within an open session.

        Updates Command/Stats/BufferDescriptor/Data regions in-place
        from the new compiled result, then signals CMD_READY.
        """
        if self._shm is None:
            raise BackendError("no active session (call open_session first)")

        layout = compiled.shm_layout
        if layout is None:
            raise BackendError("CompiledResult missing shm_layout for session batch")

        shm_image = compiled.shm_image
        buf = self._shm.buf

        if layout.total_size > len(buf):
            # Dynamic resize: ftruncate POSIX SHM, then re-mmap on Python side.
            # The C bridge will detect the size change via fstat in vten_shm_remap().
            self._resize_shm(layout.total_size)
            buf = self._shm.buf  # refreshed after resize

        # Overwrite entire SHM image (simpler and correct for resized case too)
        buf[:len(shm_image)] = shm_image

        # Reset statuses
        self._write_shm_u32(self.BACKEND_STATUS_OFFSET, BACKEND_STATUS_IDLE)
        self._write_shm_u32(self.HOST_STATUS_OFFSET, HOST_STATUS_CMD_READY)
        self._sem_h2b.post()
        logger.log(5, "[session] batch submitted (cmds=%d, bufs=%d)",
                     layout.num_commands, layout.num_buffers)

    def wait_batch(self) -> BackendResult:
        """Wait for current batch to complete. Sim stays alive."""
        return self._wait_completion()

    def close_session(self) -> None:
        """Close session: SHUTDOWN → process exit → cleanup. Idempotent."""
        if not getattr(self, "_session_active", False):
            return
        self._session_active = False
        self._shutdown_sim()
        self._release_posix_resources()

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
