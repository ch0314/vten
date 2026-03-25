"""Phase C tests: VerilatorBackend — lifecycle, configuration, SimBackend inheritance.

Spec reference: 08_backend_abstraction.md §7.3
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vten.backend.base import Backend
from vten.backend.sim_base import SimBackend


# ── Helpers ──


def _verilator_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "backend": {
            "verilator": {
                "verilator_path": "/usr/bin/verilator",
                "threads": 8,
                "trace": True,
                "opt_level": 2,
                "extra_args": ["--timing"],
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# §1  VerilatorBackend initialization
# ═══════════════════════════════════════════════════════════════════


class TestVerilatorBackendInit:
    """VerilatorBackend constructor and configuration."""

    def test_constructor_stores_config(self):
        from vten.backend.verilator import VerilatorBackend
        backend = VerilatorBackend(project_config=_verilator_config())
        assert backend._verilator_path == "/usr/bin/verilator"
        assert backend._threads == 8
        assert backend._trace is True

    def test_is_backend_subclass(self):
        from vten.backend.verilator import VerilatorBackend
        assert issubclass(VerilatorBackend, Backend)

    def test_is_sim_backend_subclass(self):
        from vten.backend.verilator import VerilatorBackend
        assert issubclass(VerilatorBackend, SimBackend)

    def test_default_config(self):
        """Defaults when no verilator config is provided."""
        from vten.backend.verilator import VerilatorBackend
        backend = VerilatorBackend(project_config={"backend": {}})
        assert backend._verilator_path == ""
        assert backend._threads == 4
        assert backend._trace is False

    def test_extra_args(self):
        from vten.backend.verilator import VerilatorBackend
        backend = VerilatorBackend(project_config=_verilator_config())
        assert backend._extra_args == ["--timing"]

    def test_cleanup_idempotent(self):
        from vten.backend.verilator import VerilatorBackend
        backend = VerilatorBackend(project_config=_verilator_config())
        backend.cleanup()
        backend.cleanup()  # Should not raise

    def test_context_manager(self):
        from vten.backend.verilator import VerilatorBackend
        with VerilatorBackend(project_config=_verilator_config()) as b:
            assert isinstance(b, Backend)

    def test_has_start_simulator(self):
        from vten.backend.verilator import VerilatorBackend
        assert hasattr(VerilatorBackend, "_start_simulator")

    def test_inherits_execute(self):
        """execute() is inherited from SimBackend, not overridden."""
        from vten.backend.verilator import VerilatorBackend
        assert VerilatorBackend.execute is SimBackend.execute

    def test_inherits_submit_wait(self):
        from vten.backend.verilator import VerilatorBackend
        assert VerilatorBackend.submit is SimBackend.submit
        assert VerilatorBackend.wait is SimBackend.wait

    def test_session_id_none_initially(self):
        from vten.backend.verilator import VerilatorBackend
        backend = VerilatorBackend(project_config=_verilator_config())
        assert backend._session_id is None
        assert backend._process is None


# ═══════════════════════════════════════════════════════════════════
# §2  Registry integration
# ═══════════════════════════════════════════════════════════════════


class TestVerilatorBackendRegistry:
    """VerilatorBackend is discoverable via backend registry."""

    def test_registry_has_verilator(self):
        from vten.backend.registry import available_backends
        assert "verilator" in available_backends()

    def test_get_backend_verilator(self):
        from vten.backend.registry import get_backend
        from vten.backend.verilator import VerilatorBackend

        backend = get_backend("verilator", _verilator_config())
        assert isinstance(backend, VerilatorBackend)

    def test_get_build_pipeline_verilator(self):
        from pathlib import Path
        from vten.backend.registry import get_build_pipeline
        from vten.build.verilator_build import VerilatorBuildPipeline

        pipeline = get_build_pipeline("verilator", Path("/tmp"), _verilator_config())
        assert isinstance(pipeline, VerilatorBuildPipeline)
