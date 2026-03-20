"""Backend base: ABC, error codes, result types.

Spec reference: 00_data_models.md §10.13, §13, 06_codegen_and_cli.md §5
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import IntEnum

from vten.errors import BackendError, BFMError, PollTimeoutError
from vten.errors import TimeoutError as VTenTimeoutError


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

    def read_buffer(self, buffer_id: int) -> bytes:
        """Read from SHM Data Region (only after DONE)."""
        return b""


# ── BatchResult — 00_data_models.md §13 ──


@dataclass
class BatchResult:
    status: str
    total_cycles: int
    per_command_stats: list[CmdStats]
    error: BackendError | None = None


# ── Backend ABC ──


class Backend(abc.ABC):
    """Abstract backend interface."""

    @abc.abstractmethod
    def submit(self, shm_image: bytes, bfm_configs: list) -> None:
        ...

    @abc.abstractmethod
    def wait(self) -> BackendResult:
        ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        ...

    @abc.abstractmethod
    def cleanup(self) -> None:
        ...
