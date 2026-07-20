"""VerilatorBackend: Verilator compiled binary backend adapter.

Extends SimBackend with Verilator-specific process management.
SHM handshake protocol is handled by the SimBackend base class.

Spec reference: 08_backend_abstraction.md §7.3
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from collections import deque

from vten.backend.sim.base import SimBackend

logger = logging.getLogger(__name__)
_sim_logger = logging.getLogger("vten.sim.verilator")


class VerilatorBackend(SimBackend):
    """Backend adapter for Verilator compiled simulation.

    Manages Verilator binary subprocess lifecycle. All SHM handshake logic
    is inherited from SimBackend.

    Unlike xsim, Verilator produces a standalone C++ binary (Vtb_top)
    that is launched directly with +plusarg flags.
    """

    def __init__(self, project_config: dict) -> None:
        super().__init__(project_config, backend_section="verilator")
        veri_cfg = project_config.get("backend", {}).get("verilator", {})
        self._verilator_path = veri_cfg.get("verilator_path", "")
        self._threads = veri_cfg.get("threads", 4)
        self._trace = veri_cfg.get("trace", False)
        self._extra_args = veri_cfg.get("extra_args", [])
        # Rolling tail of simulator output (both pipes), kept for error
        # diagnostics — the streaming threads consume the pipes themselves.
        self._output_tail: deque[str] = deque(maxlen=200)
        self._stream_threads: list[threading.Thread] = []

    def _start_simulator(self) -> None:
        """Launch Verilator compiled binary with session_id plusarg.

        The binary (Vtb_top) is expected in the kernel build directory,
        produced by the VerilatorBuildPipeline make stage.
        """
        # Locate binary: obj_dir/Vtb_top (Verilator output) or root Vtb_top
        kernel_build = self._run_ctx.kernel_build_dir
        if kernel_build:
            kb = str(kernel_build)
            obj_dir_binary = os.path.join(kb, "obj_dir", "Vtb_top")
            root_binary = os.path.join(kb, "Vtb_top")
            binary = obj_dir_binary if os.path.exists(obj_dir_binary) else root_binary
        else:
            veri_cfg = self._config.get("backend", {}).get("verilator", {})
            sim_dir = veri_cfg.get("sim_dir", str(self._run_ctx.project_dir))
            binary = os.path.join(sim_dir, "Vtb_top")

        # Resolve relative paths against project dir
        project_dir = str(self._run_ctx.project_dir)
        if not os.path.isabs(binary):
            binary = os.path.normpath(os.path.join(project_dir, binary))

        # Working directory: kernel build dir or project dir
        cwd = str(kernel_build) if kernel_build else project_dir

        cmd = [
            binary,
            f"+SESSION_ID={self._session_id}",
            f"+TIMEOUT_MS={self._timeout_ms}",
        ]

        if self._trace:
            cmd.append("+trace")

        logger.info("launching verilator: %s", " ".join(cmd))
        logger.debug("verilator cwd: %s", cwd)

        env = os.environ.copy()
        if self._run_ctx.mismatch_dir:
            env["VTEN_MISMATCH_DIR"] = str(self._run_ctx.mismatch_dir)
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            from vten.errors import BackendError
            raise BackendError(
                f"simulator binary not found: {binary}\n"
                f"Build the testbench first with 'vten build'"
            )

        # Continuously drain BOTH pipes (mirrors XsimBackend's stdout
        # streaming). An undrained PIPE fills its ~64KB kernel buffer and
        # then BLOCKS the simulator mid-$display/fprintf, wedging the whole
        # session — first hit by multi-config sessions whose accumulated
        # bridge/BFM output crossed the buffer size at the 4th batch.
        self._output_tail.clear()
        sim_verbose = self._run_ctx.sim_verbose
        self._stream_threads = []
        for stream_name, pipe in (
            ("stdout", self._process.stdout),
            ("stderr", self._process.stderr),
        ):
            t = threading.Thread(
                target=self._stream_pipe,
                args=(pipe, stream_name, sim_verbose, self._output_tail),
                daemon=True,
            )
            t.start()
            self._stream_threads.append(t)

    @staticmethod
    def _stream_pipe(pipe, stream_name: str, verbose: bool, tail: deque) -> None:
        """Drain one simulator pipe line-by-line.

        Every line is retained in the bounded ``tail`` for diagnostics;
        verbose mode additionally routes lines through Python logging.
        """
        try:
            for raw_line in pipe:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                tail.append(f"[{stream_name}] {line}")
                if verbose:
                    _sim_logger.debug("%s", line)
        except (OSError, ValueError):
            pass  # pipe closed

    def _drain_simulator_output(self) -> str:
        """Return the retained output tail (pipes are consumed by threads)."""
        for t in self._stream_threads:
            t.join(timeout=2)
        return "\n".join(self._output_tail)
