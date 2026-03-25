"""XsimBackend: Vivado xsim backend adapter.

Extends SimBackend with xsim-specific process management.
SHM handshake protocol is handled by the SimBackend base class.

Spec reference: 04_backend_xsim.md §1-6, 06_codegen_and_cli.md §4.4,
                08_backend_abstraction.md §5.5
"""

from __future__ import annotations

import os
import subprocess

from vten.backend.sim_base import SimBackend


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

        if self._config.get("_gui"):
            cmd.append("--gui")
        else:
            cmd.extend(["--runall", "--onerror", "quit"])

        self._process = subprocess.Popen(
            cmd,
            cwd=xsim_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
