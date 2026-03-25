"""Phase B tests: XrtBackend — lifecycle, configuration, error handling.

Spec reference: 08_backend_abstraction.md §6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vten.backend.base import Backend


# ── Helpers ──


def _xrt_config() -> dict:
    return {
        "project": {"name": "test_proj", "version": "0.1.0"},
        "backend": {
            "xrt": {
                "xclbin_path": "build/kernel.xclbin",
                "device_index": 0,
                "kernel_name": "conv3d",
                "poll_timeout_ms": 30000,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# §1  XrtBackend initialization
# ═══════════════════════════════════════════════════════════════════


class TestXrtBackendInit:
    """XrtBackend constructor and configuration."""

    def test_constructor_stores_config(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        assert backend._xclbin_path == "build/kernel.xclbin"
        assert backend._device_index == 0
        assert backend._kernel_name == "conv3d"

    def test_is_backend_subclass(self):
        from vten.backend.xrt import XrtBackend
        assert issubclass(XrtBackend, Backend)

    def test_lazy_init(self):
        """Device is not initialized until execute()."""
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        assert backend._device is None
        assert backend._kernel is None

    def test_default_poll_timeout(self):
        from vten.backend.xrt import XrtBackend
        config = _xrt_config()
        del config["backend"]["xrt"]["poll_timeout_ms"]
        backend = XrtBackend(project_config=config)
        assert backend._poll_timeout_ms == 30000

    def test_cleanup_idempotent(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        backend.cleanup()
        backend.cleanup()  # Should not raise

    def test_context_manager(self):
        from vten.backend.xrt import XrtBackend
        with XrtBackend(project_config=_xrt_config()) as b:
            assert isinstance(b, Backend)

    def test_execute_without_pyxrt_raises(self):
        """execute() raises BackendError when pyxrt is not available."""
        from vten.backend.xrt import XrtBackend
        from vten.errors import BackendError

        backend = XrtBackend(project_config=_xrt_config())
        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}

        with patch.dict("sys.modules", {"pyxrt": None}):
            with pytest.raises((BackendError, ImportError, ModuleNotFoundError)):
                backend.execute(compiled)


# ═══════════════════════════════════════════════════════════════════
# §2  XrtBackend execute with mock
# ═══════════════════════════════════════════════════════════════════


class TestXrtBackendExecute:
    """XrtBackend execute with mocked pyxrt."""

    def test_execute_returns_backend_result(self):
        from vten.backend.base import BackendResult
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(project_config=_xrt_config())

        # Mock internal init to skip pyxrt
        backend._device = MagicMock()
        backend._kernel = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None

        result = backend.execute(compiled)
        assert isinstance(result, BackendResult)
        assert result.status == 0

    def test_execute_empty_commands(self):
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(project_config=_xrt_config())
        backend._device = MagicMock()
        backend._kernel = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None

        result = backend.execute(compiled)
        assert result.output_buffers == {}

    def test_cleanup_after_execute(self):
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(project_config=_xrt_config())
        backend._device = MagicMock()
        backend._kernel = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None

        backend.execute(compiled)
        backend.cleanup()
        assert backend._device is None
        assert backend._kernel is None


# ═══════════════════════════════════════════════════════════════════
# §3  Registry integration
# ═══════════════════════════════════════════════════════════════════


class TestXrtBackendRegistry:
    """XrtBackend is discoverable via backend registry."""

    def test_registry_has_xrt(self):
        from vten.backend.registry import available_backends
        assert "xrt" in available_backends()

    def test_get_backend_xrt(self):
        from vten.backend.registry import get_backend
        from vten.backend.xrt import XrtBackend

        backend = get_backend("xrt", _xrt_config())
        assert isinstance(backend, XrtBackend)

    def test_get_build_pipeline_xrt(self):
        from pathlib import Path
        from vten.backend.registry import get_build_pipeline
        from vten.build.xrt_build import XrtBuildPipeline

        pipeline = get_build_pipeline("xrt", Path("/tmp"), _xrt_config())
        assert isinstance(pipeline, XrtBuildPipeline)
