"""vTen logging configuration.

Provides setup_logging() for CLI entry and module-level logger convention.
All vten modules use: logger = logging.getLogger(__name__)

Shared formatting helpers (format_elapsed, format_size) are used by both
sim and XRT backends for consistent output.
"""

from __future__ import annotations

import logging
import os
import sys


# ── Shared formatting helpers ──


def format_elapsed(seconds: float) -> str:
    """Format elapsed time: seconds for <60s, m:ss for >=60s."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}m{secs:02d}s"


def format_size(nbytes: int) -> str:
    """Format byte count as human-readable size."""
    if nbytes < 1024:
        return f"{nbytes}B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f}KB"
    else:
        return f"{nbytes / (1024 * 1024):.1f}MB"


# ── Execution phase constants (shared between sim and XRT backends) ──

PHASE_CONFIGURE = "configure"
PHASE_SEND = "send"
PHASE_TRIGGER = "trigger"
PHASE_POLL = "poll"
PHASE_RECV = "recv"
PHASE_OTHER = "other"


# ── Short component names for console output ──

_COMPONENT_MAP: dict[str, str] = {
    "vten.cli.run": "runner",
    "vten.cli.build": "cli.build",
    "vten.cli.main": "cli",
    "vten.runtime.engine": "engine",
    "vten.runtime.context": "context",
    "vten.backend.sim_base": "backend",
    "vten.backend.xsim": "xsim",
    "vten.xsim": "sim",
    "vten.backend.verilator": "verilator",
    "vten.backend.xrt": "xrt",
    "vten.build.base": "build",
    "vten.build.xsim_build": "build.xsim",
    "vten.build.xrt_build": "build.xrt",
    "vten.build.verilator_build": "build.veri",
    "vten.inference": "inference",
    "vten.backend.xrt_interpreter": "interp",
}

_COMPONENT_WIDTH = max(len(v) for v in _COMPONENT_MAP.values())

# ── ANSI colors ──

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}

_LEVEL_TAGS: dict[int, str] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: " INFO",
    logging.WARNING: " WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: " CRIT",
}


class _ConsoleFormatter(logging.Formatter):
    """Colorized formatter with short component names and aligned columns."""

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Resolve short component name
        short = _COMPONENT_MAP.get(record.name)
        if short is None:
            short = record.name.removeprefix("vten.").rsplit(".", 1)[-1]

        tag = _LEVEL_TAGS.get(record.levelno, record.levelname[:5].rjust(5))
        comp = short.ljust(_COMPONENT_WIDTH)
        msg = record.getMessage()

        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            line = f"{color}{tag}{_RESET} {_DIM}{comp}{_RESET} {_BOLD}|{_RESET} {msg}"
        else:
            line = f"{tag} {comp} | {msg}"

        # Append exception info if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line = line + "\n" + record.exc_text

        return line


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    """Configure the root 'vten' logger.

    Called once from CLI main(). Library users who import vten
    programmatically can call this or configure logging themselves.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path for a debug-level log file.
    """
    root = logging.getLogger("vten")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    # Detect color support: tty + not explicitly disabled
    use_color = (
        hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        and os.environ.get("NO_COLOR") is None
    )

    # Console handler → stderr (stdout reserved for report output)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG)
    console.setFormatter(_ConsoleFormatter(use_color=use_color))
    root.addHandler(console)

    # Optional file handler with timestamps (full module paths for grep-ability)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))
        root.addHandler(fh)
