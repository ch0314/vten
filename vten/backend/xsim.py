"""XsimBackend: Vivado xsim backend adapter.

Extends SimBackend with xsim-specific process management.
SHM handshake protocol is handled by the SimBackend base class.

Spec reference: 04_backend_xsim.md §1-6, 06_codegen_and_cli.md §4.4,
                08_backend_abstraction.md §5.5
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading

from vten.backend.sim.base import SimBackend

logger = logging.getLogger(__name__)

# Logger for SV $display output, routed through Python logging
_sim_logger = logging.getLogger("vten.xsim")

# Prefixes from SV $display that we recognize as vten BFM output
_SV_PREFIXES = ("[CTRL", "[SCHED", "[AXI4S", "[AXI4 ", "[AXILITE", "[PROBE")


def _is_suppressed(line: str) -> bool:
    """Return True if this line is xsim/Vivado boilerplate to suppress."""
    stripped = line.lstrip()
    # xsim banner and build info
    if stripped.startswith(("******", "****", "**")):
        return True
    # xsim tcl echo and control lines
    if stripped.startswith(("source xsim.dir/", "# xsim ", "run -all", "exit")):
        return True
    # Vivado info messages
    if stripped.startswith(("Time resolution is", "Info:", "INFO: [Common")):
        return True
    # Vivado process/scope debug (very long lines)
    if stripped.startswith("Time:") and "Process:" in line:
        return True
    # $finish message
    if "$finish called at time" in line:
        return True
    # DPI-C bridge internal messages
    if stripped.startswith("[vten_shm_bridge]"):
        return True
    # CTRL state transition noise — keep LOAD_BATCH, batch done, and ERROR
    if stripped.startswith("[CTRL"):
        if "LOAD_BATCH" in line or "batch #" in line or "ERROR" in line:
            return False
        return True
    return False


class XsimBackend(SimBackend):
    """Backend adapter for Vivado xsim simulator.

    Manages xsim subprocess lifecycle. All SHM handshake logic
    is inherited from SimBackend.

    Handshake protocol (04_backend_xsim.md §3):
    1. Host: shm_open, write image, sem_open, start xsim
    2. Backend (DPI-C): vten_shm_init → sem_post(b2h) "ready"
    3. Host: sem_wait(b2h), write CMD_READY, sem_post(h2b)
    4. Backend: execute → sem_post(b2h) "done/error"
    5. Host: sem_wait(b2h), read result, send ACK
    6. Shutdown: host_status=SHUTDOWN, sem_post(h2b)
    """

    def __init__(self, project_config: dict) -> None:
        super().__init__(project_config, backend_section="xsim")
        xsim_cfg = project_config.get("backend", {}).get("xsim", {})
        self._vivado_path = xsim_cfg.get("vivado_path", "")

    def _start_simulator(self) -> None:
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

        snapshot = "tb_top"  # Stage 5 uses --snapshot tb_top

        # xsim working directory: kernel build dir > xsim_dir config > project dir
        kernel_build = self._config.get("_kernel_build_dir")
        if kernel_build:
            xsim_cwd = kernel_build
        else:
            xsim_cfg = self._config.get("backend", {}).get("xsim", {})
            xsim_cwd = self._config.get("_xsim_dir",
                           xsim_cfg.get("xsim_dir",
                               self._config.get("_project_dir", ".")))
        # Resolve relative paths against project dir
        project_dir = self._config.get("_project_dir", ".")
        if not os.path.isabs(xsim_cwd):
            xsim_cwd = os.path.normpath(os.path.join(project_dir, xsim_cwd))

        cmd = [
            xsim_bin, snapshot,
            "--testplusarg", f"SESSION_ID={self._session_id}",
            "--testplusarg", f"TIMEOUT_MS={self._timeout_ms}",
        ]

        if self._config.get("_sim_verbose"):
            cmd.extend(["--testplusarg", "VTEN_VERBOSE"])

        # Probe golden buffer ID plusargs for passive probe BFMs
        probe_map = getattr(self, "_probe_buffer_map", {})
        for probe_idx, buf_id in probe_map.items():
            cmd.extend(["--testplusarg", f"PROBE_GOLDEN_{probe_idx}={buf_id}"])

        if self._config.get("_gui"):
            cmd.append("--gui")
            logger.info("xsim GUI opened. In Tcl console:")
            logger.info("  run all     — start simulation")
            logger.info("  run 1000ns  — step forward")
            if self._config.get("_waveform"):
                logger.info("  source generated/waveform.tcl  — add waveform signals")
            logger.info("Probe mismatches will pause with $stop.")
        elif self._config.get("_waveform"):
            # Batch waveform: use TCL script for log_wave + run all
            tcl_path = os.path.join(xsim_cwd, "generated", "waveform.tcl")
            if os.path.isfile(tcl_path):
                cmd.extend(["--tclbatch", tcl_path])
            else:
                logger.warning("waveform.tcl not found at %s, falling back to --runall", tcl_path)
                cmd.extend(["--runall", "--onerror", "quit"])
        else:
            cmd.extend(["--runall", "--onerror", "quit"])

        logger.info("launching xsim: %s", " ".join(cmd))
        logger.log(5, "xsim cwd: %s", xsim_cwd)

        # Always capture stdout via PIPE so we can route through Python logging
        # Build env with optional VTEN_MISMATCH_DIR for probe mismatch logging
        env = os.environ.copy()
        mismatch_dir = self._config.get("_mismatch_dir")
        if mismatch_dir:
            env["VTEN_MISMATCH_DIR"] = str(mismatch_dir)
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=xsim_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            from vten.errors import BackendError
            if not os.path.isdir(xsim_cwd):
                raise BackendError(
                    f"build directory not found: {xsim_cwd}\n"
                    f"Run 'vten build' first to compile the simulation snapshot"
                )
            raise BackendError(
                f"xsim not found: {xsim_bin}\n"
                f"Check that Vivado is installed and "
                f"vivado_path is set in vten.toml [backend.xsim]"
            )

        # Stream xsim stdout through Python logger in a background thread
        sim_verbose = self._config.get("_sim_verbose")
        self._stdout_thread = threading.Thread(
            target=self._stream_stdout,
            args=(self._process.stdout, sim_verbose),
            daemon=True,
        )
        self._stdout_thread.start()

    @staticmethod
    def _stream_stdout(pipe, verbose: bool) -> None:
        """Read xsim stdout line-by-line and route through Python logging.

        In verbose mode: vten BFM lines are logged at DEBUG level,
        xsim/Vivado boilerplate is suppressed.
        In non-verbose mode: lines are silently consumed.
        """
        try:
            for raw_line in pipe:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                if not verbose:
                    continue
                if _is_suppressed(line):
                    continue
                # vten BFM output
                if any(line.startswith(p) for p in _SV_PREFIXES):
                    _sim_logger.debug("%s", line)
                elif "ERROR" in line or "$stop" in line:
                    _sim_logger.warning("%s", line)
                else:
                    # DUT $display or other — still show
                    _sim_logger.debug("%s", line)
        except (OSError, ValueError):
            pass  # pipe closed
