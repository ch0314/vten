"""XrtBackend — XRT/pyxrt-based FPGA backend.

Executes IR Commands on real FPGA hardware via Xilinx Runtime (XRT).
Uses CommandInterpreter to translate IR Commands to XRT API calls.

Spec reference: 08_backend_abstraction.md §6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vten.backend.base import Backend, BackendResult
from vten.errors import BackendError

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


class XrtBackend(Backend):
    """XRT/pyxrt-based FPGA backend.

    vten.toml [backend.xrt] configuration:
      xclbin_path:      Path to xclbin file
      device_index:     FPGA device index (default: 0)
      kernel_name:      Kernel name within xclbin
      poll_timeout_ms:  POLL_REG timeout in ms (default: 30000)
    """

    def __init__(self, project_config: dict) -> None:
        xrt_cfg = project_config.get("backend", {}).get("xrt", {})
        self._xclbin_path = xrt_cfg.get("xclbin_path", "")
        self._device_index = xrt_cfg.get("device_index", 0)
        self._kernel_name = xrt_cfg.get("kernel_name", "")
        self._poll_timeout_ms = xrt_cfg.get("poll_timeout_ms", 30000)
        self._config = project_config

        # XRT resources (lazy init on first execute)
        self._device = None
        self._xclbin = None
        self._kernel = None
        self._xrt = None  # pyxrt module
        self._interpreter = None

    def _init_device(self) -> None:
        """Initialize FPGA device, load xclbin, create kernel object.

        Called lazily on first execute() to allow configuration
        without requiring actual FPGA hardware.
        """
        try:
            import pyxrt
        except ImportError as e:
            raise BackendError(
                "pyxrt not available. Install XRT runtime to use XRT backend. "
                "See: https://github.com/Xilinx/XRT"
            ) from e

        self._xrt = pyxrt

        if not self._xclbin_path:
            raise BackendError(
                "xclbin_path not configured in [backend.xrt]. "
                "Build xclbin first: v++ --link ..."
            )

        self._device = pyxrt.device(self._device_index)
        self._xclbin = pyxrt.xclbin(self._xclbin_path)
        self._device.load_xclbin(self._xclbin)

        kernel_name = self._kernel_name
        if not kernel_name:
            # Try to auto-detect from xclbin
            kernels = self._xclbin.get_kernels()
            if kernels:
                kernel_name = kernels[0].get_name()
            else:
                raise BackendError(
                    "kernel_name not configured and could not be "
                    "auto-detected from xclbin"
                )

        self._kernel = pyxrt.kernel(
            self._device, self._xclbin.get_uuid(), kernel_name,
        )

    def _build_arg_map(self, compiled: CompiledResult) -> dict[int, int]:
        """Build buffer_id → kernel arg_index mapping from compiled result.

        Uses XRT interface config from kernel spec (xrt.arg_index/arg_name)
        or falls back to sequential assignment.
        """
        arg_map: dict[int, int] = {}

        if compiled.flattened_view is None:
            return arg_map

        auto_index = 0
        for name, exposed in compiled.flattened_view.exposed_tensors.items():
            buffer_id = compiled.buffer_ids.get(name)
            if buffer_id is None:
                continue

            # Try to get XRT arg config from interface spec
            try:
                iface = compiled.flattened_view.top_spec.get_interface(
                    exposed.top_interface
                )
                if iface.xrt and iface.xrt.arg_index is not None:
                    arg_map[buffer_id] = iface.xrt.arg_index
                    continue
            except (KeyError, AttributeError):
                pass

            # Fallback: sequential assignment
            arg_map[buffer_id] = auto_index
            auto_index += 1

        return arg_map

    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Execute compiled result on FPGA via XRT.

        1. Initialize device if needed (lazy)
        2. Build buffer_id → arg_index mapping
        3. Execute IR commands via CommandInterpreter
        4. Return output buffers in BackendResult
        """
        if self._device is None:
            self._init_device()

        from vten.runtime.interpreter import CommandInterpreter

        arg_map = self._build_arg_map(compiled)
        interpreter = CommandInterpreter(
            device=self._device,
            kernel=self._kernel,
            xrt_module=self._xrt,
            arg_map=arg_map,
            poll_timeout_ms=self._poll_timeout_ms,
        )
        self._interpreter = interpreter

        interpreter.execute(compiled.commands, compiled.tensor_data)

        return BackendResult(
            status=0,
            output_buffers=interpreter.output_buffers,
        )

    def cleanup(self) -> None:
        """Release XRT resources. Idempotent."""
        if self._interpreter is not None:
            self._interpreter.cleanup()
            self._interpreter = None
        self._kernel = None
        self._device = None
        self._xclbin = None
