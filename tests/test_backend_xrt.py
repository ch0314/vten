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
        assert backend._default_ip is None
        assert backend._ips == {}

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

        with patch.dict("sys.modules", {"pyxrt": None, "vten_xrt": None}):
            with pytest.raises((BackendError, ImportError, ModuleNotFoundError)):
                backend.execute(compiled)

    def test_instance_name_config(self):
        from vten.backend.xrt import XrtBackend
        config = _xrt_config()
        config["backend"]["xrt"]["instance_name"] = "conv3d_1"
        backend = XrtBackend(project_config=config)
        assert backend._instance_name == "conv3d_1"

    def test_compile_target_is_hw(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        assert backend.compile_target == "hw"


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
        backend._default_ip = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None
        compiled.iface_id_to_name = {}
        compiled.buffer_ids = {}

        result = backend.execute(compiled)
        assert isinstance(result, BackendResult)
        assert result.status == 0

    def test_execute_empty_commands(self):
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(project_config=_xrt_config())
        backend._device = MagicMock()
        backend._default_ip = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None
        compiled.iface_id_to_name = {}
        compiled.buffer_ids = {}

        result = backend.execute(compiled)
        assert result.output_buffers == {}

    def test_cleanup_after_execute(self):
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(project_config=_xrt_config())
        backend._device = MagicMock()
        backend._default_ip = MagicMock()
        backend._xrt = MagicMock()

        compiled = MagicMock()
        compiled.commands = []
        compiled.tensor_data = {}
        compiled.flattened_view = None
        compiled.iface_id_to_name = {}
        compiled.buffer_ids = {}

        backend.execute(compiled)
        backend.cleanup()
        assert backend._device is None
        assert backend._default_ip is None
        assert backend._ips == {}


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


# ═══════════════════════════════════════════════════════════════════
# §4  XrtBackend builder methods
# ═══════════════════════════════════════════════════════════════════


class TestXrtBackendBuilders:
    """Test _build_ip_map, _build_mem_bank_map, _build_addr_bindings."""

    def test_build_ip_map_empty_when_no_view(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        compiled = MagicMock()
        compiled.flattened_view = None
        compiled.iface_id_to_name = {}
        assert backend._build_ip_map(compiled) == {}

    def test_build_mem_bank_map_empty_when_no_view(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        compiled = MagicMock()
        compiled.flattened_view = None
        assert backend._build_mem_bank_map(compiled) == {}

    def test_build_addr_bindings_empty_when_no_view(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        compiled = MagicMock()
        compiled.flattened_view = None
        assert backend._build_addr_bindings(compiled) == {}

    def test_build_addr_bindings_empty_when_no_bindings(self):
        from vten.backend.xrt import XrtBackend
        backend = XrtBackend(project_config=_xrt_config())
        compiled = MagicMock()
        compiled.flattened_view._register_bindings = None
        assert backend._build_addr_bindings(compiled) == {}


# ═══════════════════════════════════════════════════════════════════
# §5  CommandInterpreter multi-IP and address translation
# ═══════════════════════════════════════════════════════════════════


class TestInterpreterMultiIP:
    """CommandInterpreter dispatches to correct IP per interface_id."""

    def _make_interpreter(self, ip_map=None, addr_bindings=None,
                          mem_bank_map=None):
        from vten.runtime.interpreter import CommandInterpreter
        device = MagicMock()
        default_ip = MagicMock()
        xrt_mod = MagicMock()
        return CommandInterpreter(
            device=device,
            kernel=default_ip,
            xrt_module=xrt_mod,
            ip_map=ip_map or {},
            addr_bindings=addr_bindings or {},
            mem_bank_map=mem_bank_map or {},
        )

    def test_get_ip_returns_mapped_ip(self):
        ip_a = MagicMock(name="ip_a")
        interp = self._make_interpreter(ip_map={1: ip_a})
        assert interp._get_ip(1) is ip_a

    def test_get_ip_falls_back_to_default(self):
        interp = self._make_interpreter()
        result = interp._get_ip(99)
        assert result is interp._kernel  # fallback to default

    def test_write_reg_dispatches_to_correct_ip(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        ip_ctrl = MagicMock(name="ip_ctrl")
        interp = self._make_interpreter(ip_map={2: ip_ctrl})

        cmd = Command(
            op=OpCode.WRITE_REG,
            cmd_id=0,
            interface_id=2,
            protocol=Protocol.AXI4L,
            reg_offset=0x10,
            reg_value=42,
        )
        interp._exec_write_reg(cmd)
        ip_ctrl.write_register.assert_called_once_with(0x10, 42)

    def test_poll_reg_dispatches_to_correct_ip(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        ip_ctrl = MagicMock(name="ip_ctrl")
        ip_ctrl.read_register.return_value = 1  # immediately done
        interp = self._make_interpreter(ip_map={3: ip_ctrl})

        cmd = Command(
            op=OpCode.POLL_REG,
            cmd_id=0,
            interface_id=3,
            protocol=Protocol.AXI4L,
            reg_offset=0x54,
            reg_mask=1,
            reg_expected=1,
        )
        interp._exec_poll_reg(cmd)
        ip_ctrl.read_register.assert_called_with(0x54)

    def test_read_reg_dispatches_to_correct_ip(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        ip_ctrl = MagicMock(name="ip_ctrl")
        ip_ctrl.read_register.return_value = 0xBEEF
        interp = self._make_interpreter(ip_map={1: ip_ctrl})

        cmd = Command(
            op=OpCode.READ_REG,
            cmd_id=0,
            interface_id=1,
            protocol=Protocol.AXI4L,
            reg_offset=0x20,
        )
        interp._exec_read_reg(cmd)
        assert cmd.reg_value == 0xBEEF


class TestInterpreterAddrTranslation:
    """CommandInterpreter substitutes BO address for auto_bind registers."""

    def _make_interpreter(self, addr_bindings):
        from vten.runtime.interpreter import CommandInterpreter
        device = MagicMock()
        default_ip = MagicMock()
        xrt_mod = MagicMock()
        return CommandInterpreter(
            device=device,
            kernel=default_ip,
            xrt_module=xrt_mod,
            addr_bindings=addr_bindings,
        )

    def test_addr_substitution_full_address(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        interp = self._make_interpreter(
            addr_bindings={(1, 0x38): (5, None)},
        )

        # Simulate LOAD creating a BO
        mock_bo = MagicMock()
        mock_bo.address.return_value = 0x00000001_ABCD0000
        interp._buffers[5] = mock_bo

        cmd = Command(
            op=OpCode.WRITE_REG,
            cmd_id=0,
            interface_id=1,
            protocol=Protocol.AXI4L,
            reg_offset=0x38,
            reg_value=999,  # SHM offset (should be replaced)
        )
        interp._exec_write_reg(cmd)
        interp._kernel.write_register.assert_called_once_with(
            0x38, 0x00000001_ABCD0000
        )

    def test_addr_substitution_with_bits(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        interp = self._make_interpreter(
            addr_bindings={
                (1, 0x38): (5, "31:0"),
                (1, 0x3C): (5, "63:32"),
            },
        )

        mock_bo = MagicMock()
        mock_bo.address.return_value = 0x00000001_ABCD0000
        interp._buffers[5] = mock_bo

        cmd_lo = Command(
            op=OpCode.WRITE_REG, cmd_id=0, interface_id=1,
            protocol=Protocol.AXI4L, reg_offset=0x38, reg_value=0,
        )
        cmd_hi = Command(
            op=OpCode.WRITE_REG, cmd_id=1, interface_id=1,
            protocol=Protocol.AXI4L, reg_offset=0x3C, reg_value=0,
        )
        interp._exec_write_reg(cmd_lo)
        interp._exec_write_reg(cmd_hi)

        calls = interp._kernel.write_register.call_args_list
        assert calls[0].args == (0x38, 0xABCD0000)
        assert calls[1].args == (0x3C, 0x00000001)

    def test_no_binding_passes_through(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        interp = self._make_interpreter(addr_bindings={})

        cmd = Command(
            op=OpCode.WRITE_REG, cmd_id=0, interface_id=1,
            protocol=Protocol.AXI4L, reg_offset=0x10, reg_value=42,
        )
        interp._exec_write_reg(cmd)
        interp._kernel.write_register.assert_called_once_with(0x10, 42)

    def test_binding_but_bo_not_loaded_falls_through(self):
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode, Protocol

        interp = self._make_interpreter(
            addr_bindings={(1, 0x38): (5, None)},
        )
        # buffer_id=5 not in _buffers

        cmd = Command(
            op=OpCode.WRITE_REG, cmd_id=0, interface_id=1,
            protocol=Protocol.AXI4L, reg_offset=0x38, reg_value=999,
        )
        interp._exec_write_reg(cmd)
        interp._kernel.write_register.assert_called_once_with(0x38, 999)


class TestInterpreterMemBank:
    """CommandInterpreter uses mem_bank_map for BO allocation."""

    def test_load_uses_mem_bank_map(self):
        from vten.runtime.interpreter import CommandInterpreter
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode

        device = MagicMock()
        xrt_mod = MagicMock()
        mock_bo = MagicMock()
        xrt_mod.bo.return_value = mock_bo

        interp = CommandInterpreter(
            device=device,
            kernel=MagicMock(),
            xrt_module=xrt_mod,
            mem_bank_map={7: 33},  # buffer_id 7 → bank 33
        )

        cmd = Command(op=OpCode.LOAD, cmd_id=0, buffer_id=7)
        interp._exec_load(cmd, {7: b"\x00" * 16})

        # Verify BO was created with bank 33
        xrt_mod.bo.assert_called_once_with(
            device, 16, xrt_mod.bo.flags.normal, 33,
        )

    def test_load_defaults_to_bank_0(self):
        from vten.runtime.interpreter import CommandInterpreter
        from vten.runtime.ir import Command
        from vten.spec.models import OpCode

        device = MagicMock()
        xrt_mod = MagicMock()
        xrt_mod.bo.return_value = MagicMock()

        interp = CommandInterpreter(
            device=device,
            kernel=MagicMock(),
            xrt_module=xrt_mod,
            mem_bank_map={},  # no mapping
        )

        cmd = Command(op=OpCode.LOAD, cmd_id=0, buffer_id=3)
        interp._exec_load(cmd, {3: b"\x00" * 8})

        xrt_mod.bo.assert_called_once_with(
            device, 8, xrt_mod.bo.flags.normal, 0,
        )
