"""Backend base: ABC, error codes, result types, RunContext.

Spec reference: 00_data_models.md §10.13, §13, 06_codegen_and_cli.md §5,
                08_backend_abstraction.md §5
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from vten.errors import BackendError, BFMError, PollTimeoutError, ProbeMismatchError
from vten.errors import TimeoutError as VTenTimeoutError

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


# ── RunContext — runtime execution state (replaces _ prefix keys) ──


@dataclass
class RunContext:
    """Runtime execution context passed to backends and pipelines.

    Replaces the ``_`` prefix keys that were previously injected into the
    project config dict (e.g. ``config["_project_dir"]``).  This gives
    type-safe access and keeps the user-facing config dict clean.
    """

    project_dir: Path = field(default_factory=lambda: Path("."))
    kernel_build_dir: Path | None = None
    waveform: bool = False
    waveform_on_fail: bool = False
    gui: bool = False
    sim_verbose: bool = False
    mismatch_dir: Path | None = None


# ── CompileTarget — backend type identifier ──


class CompileTarget(Enum):
    SIM = "sim"
    HW = "hw"

    def __eq__(self, other: object) -> bool:
        """Support string comparison for backward compatibility."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value)


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
    BackendErrorCode.PROBE_MISMATCH: ProbeMismatchError,
    BackendErrorCode.TIMEOUT: VTenTimeoutError,
}


def raise_backend_error(code: int, cmd_id: int, message: str) -> None:
    """Raise appropriate exception for a backend error code."""
    exc_cls = _ERROR_MAP.get(code, BackendError)
    full_message = f"{message} (cmd_id={cmd_id})"
    kwargs: dict = {"context": {"error_code": code, "cmd_id": cmd_id}}
    if issubclass(exc_cls, ProbeMismatchError):
        kwargs["cmd_id"] = cmd_id
    raise exc_cls(full_message, **kwargs)


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
    SIM backends may additionally override shutdown() for process signalling.
    SimBackend auto-manages session lifecycle within execute().

    Subclasses store ``_run_ctx`` for typed access to runtime state
    (project_dir, waveform flags, etc.).  The context is set via
    ``set_run_context()`` or by the ``run_ctx`` constructor kwarg
    on concrete backends.
    """

    _run_ctx: RunContext

    def set_run_context(self, ctx: RunContext) -> None:
        """Attach (or replace) the runtime execution context."""
        self._run_ctx = ctx

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
        """Close active session (if any) and release resources. Idempotent."""
        ...

    def __enter__(self) -> Backend:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()

    # ── Optional lifecycle control ──

    def shutdown(self) -> None:
        """Send shutdown signal (optional)."""
        pass

    def get_buffer_object(self, buffer_id: int) -> object | None:
        """Return device buffer object for buffer_id, or None.

        Used by inference API to bind output BOs to Tensor objects.
        Default returns None (SIM backends don't expose device buffers).
        """
        return None

    def inject_prebound(self, buffer_id: int, bo: object) -> None:
        """Inject a pre-existing device buffer for reuse.

        Used by inference API for upload() — BO already on device,
        skip LOAD+PUSH. No-op by default (SIM backends).
        """
        pass

    def working_directory(self, kernel_dir: Path, project_dir: Path) -> Path:
        """Return CWD for test execution. Default: project root."""
        return project_dir

    @property
    def compile_target(self) -> CompileTarget:
        """Compile target for backend type identification.

        Returns CompileTarget.SIM for simulation backends,
        CompileTarget.HW for hardware backends.
        """
        return CompileTarget.SIM

