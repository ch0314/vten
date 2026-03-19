"""Verilator simulation harness for Phase 3 functional testing.

Communicates with compiled verilator drivers via subprocess stdin/stdout
using a simple JSON line protocol. Each module has its own driver executable.

Usage:
    sim = VerilatorSim(driver_path)
    sim.load_shm(image_path)
    sim.create()
    sim.reset(5)
    result = sim.tick(1)
    assert result['state_name'] == 'S_FEED'
    sim.destroy()
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class VerilatorSim:
    """Subprocess-based verilator simulation driver."""

    def __init__(self, driver_path: str | Path) -> None:
        self.driver_path = Path(driver_path)
        if not self.driver_path.exists():
            raise FileNotFoundError(f"Driver not found: {self.driver_path}")
        self._proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [str(self.driver_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _send(self, cmd: dict[str, Any]) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(cmd) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"Driver process died. stderr: {stderr}")
        return json.loads(line.strip())

    def _check(self, result: dict[str, Any]) -> dict[str, Any]:
        if "error" in result:
            raise RuntimeError(f"Driver error: {result['error']}")
        return result

    # ── SHM image loading ──

    def load_shm_image(self, image: bytes | bytearray) -> dict[str, Any]:
        """Write SHM image to temp file and load into mock."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(bytes(image))
            path = f.name
        result = self._check(self._send({"cmd": "load", "file": path}))
        Path(path).unlink(missing_ok=True)
        return result

    def load_shm_file(self, path: str | Path) -> dict[str, Any]:
        """Load SHM image from existing file."""
        return self._check(self._send({"cmd": "load", "file": str(path)}))

    # ── Simulator lifecycle ──

    def create(self) -> dict[str, Any]:
        return self._check(self._send({"cmd": "create"}))

    def destroy(self) -> dict[str, Any]:
        return self._check(self._send({"cmd": "destroy"}))

    def reset(self, cycles: int = 5) -> dict[str, Any]:
        return self._check(self._send({"cmd": "reset", "cycles": cycles}))

    # ── Simulation ──

    def tick(self, n: int = 1) -> dict[str, Any]:
        return self._check(self._send({"cmd": "tick", "n": n}))

    def get_state(self) -> dict[str, Any]:
        return self._check(self._send({"cmd": "get_state"}))

    def get_feed(self) -> dict[str, Any]:
        return self._check(self._send({"cmd": "get_feed"}))

    def get_internals(self) -> dict[str, Any]:
        return self._check(self._send({"cmd": "get_internals"}))

    # ── Signal control ──

    def set_signal(self, signal: str, value: int) -> None:
        self._check(self._send({"cmd": "set", "signal": signal, "value": value}))

    def mock_set(self, field: str, value: int) -> None:
        self._check(self._send({"cmd": "mock_set", "field": field, "value": value}))

    def mock_get(self, field: str) -> int:
        result = self._check(self._send({"cmd": "mock_get", "field": field}))
        return result["value"]

    # ── Convenience ──

    @property
    def state_name(self) -> str:
        return self.get_state()["state_name"]

    @property
    def state(self) -> int:
        return self.get_state()["state"]

    # ── Cleanup ──

    def quit(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"cmd": "quit"})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self) -> None:
        self.quit()

    def __enter__(self) -> VerilatorSim:
        return self

    def __exit__(self, *args: Any) -> None:
        self.quit()
