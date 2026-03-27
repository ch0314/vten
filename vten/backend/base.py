"""Backend base: ABC, error codes, result types.

Spec reference: 00_data_models.md §10.13, §13, 06_codegen_and_cli.md §5,
                08_backend_abstraction.md §5
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from vten.errors import BackendError, BFMError, PollTimeoutError
from vten.errors import TimeoutError as VTenTimeoutError

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


# ── BackendErrorCode — 00_data_models.md §10.13 ──


class BackendErrorCode(IntEnum):
    OK = 0
    ADDR_UNMATCH = 1
    POLL_TIMEOUT = 2
    BFM_QUEUE_ERROR = 3
    SCHEDULER_ERROR = 4
    SHM_ACCESS_ERROR = 5
    UNKNOWN_OPCODE = 6
    BFM_MAP_ERROR = 7
    PROBE_MISMATCH = 8
    TIMEOUT = 9


# ── Error code → exception mapping ──

_ERROR_MAP: dict[int, type[BackendError]] = {
    BackendErrorCode.ADDR_UNMATCH: BFMError,
    BackendErrorCode.POLL_TIMEOUT: PollTimeoutError,
    BackendErrorCode.BFM_QUEUE_ERROR: BFMError,
    BackendErrorCode.TIMEOUT: VTenTimeoutError,
}


def raise_backend_error(code: int, cmd_id: int, message: str) -> None:
    """Raise appropriate exception for a backend error code."""
    exc_cls = _ERROR_MAP.get(code, BackendError)
    full_message = f"{message} (cmd_id={cmd_id})"
    raise exc_cls(
        full_message,
        context={"error_code": code, "cmd_id": cmd_id},
    )


# ── CmdStats — Stats Region per-command metrics ──


@dataclass
class CmdStats:
    cmd_id: int
    status: int
    issue_cycle: int
    commit_cycle: int
    first_active_cycle: int
    last_active_cycle: int
    active_cycles: int
    total_beats: int
    stall_cycles: int

    @property
    def latency_cycles(self) -> int:
        return self.commit_cycle - self.issue_cycle

    @property
    def active_window(self) -> int:
        return self.last_active_cycle - self.first_active_cycle + 1

    @property
    def utilization(self) -> float:
        window = self.active_window
        if window == 0:
            return 0.0
        return self.active_cycles / window

    @property
    def bus_efficiency(self) -> float:
        latency = self.latency_cycles
        if latency == 0:
            return 0.0
        return self.active_cycles / latency


# ── BackendResult — 00_data_models.md §13 ──


@dataclass
class BackendResult:
    status: int
    error_code: int = 0
    error_cmd_id: int = 0
    error_message: str = ""
    stats: list[CmdStats] = field(default_factory=list)
    output_buffers: dict[int, bytes] = field(default_factory=dict)
    _shm_reader: Callable[[int], bytes] | None = field(
        default=None, repr=False
    )

    def read_buffer(self, buffer_id: int) -> bytes:
        """Read buffer data by buffer_id.

        Checks output_buffers first (XRT path), then falls back to
        _shm_reader closure (SIM path). Must be called before cleanup()
        destroys the SHM segment when using _shm_reader.
        Returns raw bytes for the given buffer_id, or b"" if unavailable.
        """
        if buffer_id in self.output_buffers:
            return self.output_buffers[buffer_id]
        if self._shm_reader is not None:
            return self._shm_reader(buffer_id)
        return b""


# ── BatchResult — 00_data_models.md §13 ──


@dataclass
class BatchResult:
    status: str
    total_cycles: int
    per_command_stats: list[CmdStats]
    error: BackendError | None = None


# ── Backend ABC — 08_backend_abstraction.md §5 ──


class Backend(abc.ABC):
    """Abstract backend interface.

    All backends must implement execute() and cleanup().
    SIM backends may additionally override submit()/wait()/shutdown()
    for fine-grained lifecycle control.
    """

    @abc.abstractmethod
    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Execute compiled result and return backend result.

        This is the primary entry point. Internally:
          - SIM: SHM image write → simulator start → handshake → result read
          - HW:  IR command interpretation → XRT API calls → result collect
        """
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        """Release resources. Idempotent."""
        ...

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # ── Optional fine-grained control (SimBackend overrides) ──

    def submit(self, compiled: CompiledResult) -> None:
        """Async submit (optional). First half of execute()."""
        raise NotImplementedError("Use execute() for synchronous operation")

    def wait(self) -> BackendResult:
        """Wait for result (optional). Second half of execute()."""
        raise NotImplementedError("Use execute() for synchronous operation")

    def shutdown(self) -> None:
        """Send shutdown signal (optional)."""
        pass

    @property
    def compile_target(self) -> str:
        """Compile target for RuntimeEngine.compile().

        Returns "sim" for simulation backends (includes SHM packing),
        "hw" for hardware backends (skips SHM packing).
        """
        return "sim"

    # ── Session protocol (multi-batch) ──

    @property
    def supports_session(self) -> bool:
        """Whether this backend supports persistent session mode."""
        return False

    def open_session(self, compiled: CompiledResult) -> None:
        """Open a persistent session with the first batch's compiled result.

        Creates SHM, starts simulator, waits for ready. The Data Region
        is preserved across batches for cross-batch buffer aliasing.
        """
        raise NotImplementedError

    def submit_batch(self, compiled: CompiledResult) -> None:
        """Submit a new batch within an open session.

        Updates Command, Stats, and BufferDescriptor regions in-place,
        writes new tensor data, then signals CMD_READY.
        """
        raise NotImplementedError

    def wait_batch(self) -> BackendResult:
        """Wait for the current batch to complete within a session.

        Does NOT send SHUTDOWN — the simulator stays alive.
        """
        raise NotImplementedError

    def close_session(self) -> None:
        """Close the session: send SHUTDOWN, wait for process exit, cleanup.

        Idempotent — safe to call multiple times.
        """
        pass
