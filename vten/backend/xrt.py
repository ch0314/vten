"""XrtBackend — XRT/pyxrt-based FPGA backend.

Executes IR Commands on real FPGA hardware via Xilinx Runtime (XRT).
Uses xrt.ip for raw register access (no ap_ctrl_hs assumption).
Uses CommandInterpreter to translate IR Commands to XRT API calls.

Spec reference: 08_backend_abstraction.md §6
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vten.backend.base import Backend, BackendResult
from vten.errors import BackendError

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


class XrtBackend(Backend):
    """XRT/pyxrt-based FPGA backend using xrt.ip for raw register access.

    vten.toml [backend.xrt] configuration:
      xclbin_path:      Path to xclbin file
      device_index:     FPGA device index (default: 0)
      kernel_name:      Default IP kernel name within xclbin
      instance_name:    Default IP instance name (optional)
      poll_timeout_ms:  POLL_REG timeout in ms (default: 30000)
    """

    def __init__(self, project_config: dict) -> None:
        xrt_cfg = project_config.get("backend", {}).get("xrt", {})
        self._xclbin_path = xrt_cfg.get("xclbin_path", "")
        self._device_index = xrt_cfg.get("device_index", 0)
        self._kernel_name = xrt_cfg.get("kernel_name", "")
        self._instance_name = xrt_cfg.get("instance_name", "")
        self._poll_timeout_ms = xrt_cfg.get("poll_timeout_ms", 30000)
        self._config = project_config

        # XRT resources (lazy init on first execute)
        self._device: Any = None
        self._xclbin: Any = None
        self._uuid: Any = None
        self._xrt: Any = None  # pyxrt module
        self._default_ip: Any = None
        self._group_ids: dict[int, int] = {}  # arg_index → memory group
        self._ips: dict[str, Any] = {}  # ip_name → xrt.ip
        self._interpreter: Any = None

    def _setup_emulation_env(self) -> None:
        """Auto-configure emulation environment if xclbin is hw_emu."""
        import os
        import shutil

        if not self._xclbin_path:
            return

        xclbin = Path(self._xclbin_path)
        if "_hw_emu" in xclbin.name or "hw_emu" in str(xclbin.parent):
            if not os.environ.get("XCL_EMULATION_MODE"):
                os.environ["XCL_EMULATION_MODE"] = "hw_emu"

            # Copy emconfig.json to CWD if not present
            cwd_emconfig = Path.cwd() / "emconfig.json"
            if not cwd_emconfig.exists():
                xclbin_emconfig = xclbin.parent / "emconfig.json"
                if xclbin_emconfig.exists():
                    shutil.copy2(xclbin_emconfig, cwd_emconfig)

    def _init_device(self) -> None:
        """Initialize FPGA device and load xclbin.

        Called lazily on first execute() to allow configuration
        without requiring actual FPGA hardware.
        """
        # Check XRT installation
        xrt_setup = Path("/opt/xilinx/xrt/setup.sh")
        if not xrt_setup.exists():
            import os
            if not os.environ.get("XILINX_XRT"):
                raise BackendError(
                    "XRT not found. Install Xilinx Runtime (XRT) and "
                    "source /opt/xilinx/xrt/setup.sh before running."
                )

        try:
            import vten_xrt as pyxrt  # vTen's own XRT bindings (xrt::ip support)
        except ImportError:
            try:
                import pyxrt  # vendor fallback
            except ImportError as e:
                raise BackendError(
                    "XRT Python bindings not available. "
                    "Build vten_xrt: cd vten/xrt_binding && mkdir build "
                    "&& cd build && cmake .. && make && pip install . "
                    "Or install XRT runtime: https://github.com/Xilinx/XRT"
                ) from e

        self._xrt = pyxrt

        if not self._xclbin_path:
            raise BackendError(
                "xclbin_path not configured in [backend.xrt]. "
                "Build xclbin first: vten build --backend xrt"
            )

        xclbin_file = Path(self._xclbin_path)
        if not xclbin_file.exists():
            raise BackendError(
                f"xclbin not found: {self._xclbin_path}. "
                "Build it first: vten build --backend xrt --run-vivado"
            )

        self._setup_emulation_env()

        self._device = pyxrt.device(self._device_index)
        self._xclbin = pyxrt.xclbin(self._xclbin_path)
        self._device.load_xclbin(self._xclbin)
        self._uuid = self._xclbin.get_uuid()

        # Create default kernel (for group_id queries) and IP
        if self._kernel_name:
            if self._instance_name:
                ip_name = f"{self._kernel_name}:{{{self._instance_name}}}"
            else:
                ip_name = self._kernel_name

            # Create xrt.kernel first for group_id() (memory bank queries),
            # then delete it before creating xrt.ip (exclusive access).
            # xrt.kernel and xrt.ip cannot coexist for the same CU.
            if hasattr(pyxrt, "kernel"):
                try:
                    tmp_kernel = pyxrt.kernel(
                        self._device, self._uuid, ip_name,
                    )
                    # Cache group_ids before releasing kernel
                    self._group_ids: dict[int, int] = {}
                    for arg_idx in range(16):
                        try:
                            self._group_ids[arg_idx] = tmp_kernel.group_id(arg_idx)
                        except Exception:
                            break
                    del tmp_kernel
                except Exception:
                    pass

            self._default_ip = self._get_or_create_ip(ip_name)

    def _get_or_create_ip(self, ip_name: str) -> Any:
        """Get or lazily create an xrt.ip by name."""
        if ip_name not in self._ips:
            self._ips[ip_name] = self._xrt.ip(
                self._device, self._uuid, ip_name,
            )
        return self._ips[ip_name]

    def _build_ip_map(self, compiled: CompiledResult) -> dict[int, Any]:
        """Build interface_id → xrt.ip mapping from compiled result.

        Uses xrt.ip_name from each interface's XRT config.
        Falls back to default_ip when ip_name is not specified.
        """
        ip_map: dict[int, Any] = {}

        if compiled.flattened_view is None:
            return ip_map

        for iface_id, iface_name in compiled.iface_id_to_name.items():
            try:
                iface_spec = compiled.flattened_view.top_spec.get_interface(
                    iface_name
                )
            except (KeyError, AttributeError):
                if self._default_ip is not None:
                    ip_map[iface_id] = self._default_ip
                continue

            if iface_spec.xrt and iface_spec.xrt.ip_name:
                ip_map[iface_id] = self._get_or_create_ip(
                    iface_spec.xrt.ip_name
                )
            elif self._default_ip is not None:
                ip_map[iface_id] = self._default_ip

        return ip_map

    def _build_mem_bank_map(
        self, compiled: CompiledResult,
    ) -> dict[int, int]:
        """Build buffer_id → memory bank index mapping.

        Prefers kernel.group_id() (runtime-accurate) over
        xrt.memory_bank_index from kernel_spec (may differ from XRT
        runtime group, e.g. U280 DDR[0] is group 32, not 0).

        For interfaces with xrt.arg_index, uses kernel.group_id(arg_index)
        directly. Otherwise falls back to xrt.memory_bank_index.
        """
        bank_map: dict[int, int] = {}

        if compiled.flattened_view is None:
            return bank_map

        # Use cached group_ids from init (kernel was released for ip access)
        group_ids: dict[int, int] = getattr(self, "_group_ids", {})

        for name, exposed in compiled.flattened_view.exposed_tensors.items():
            buffer_id = compiled.buffer_ids.get(name)
            if buffer_id is None:
                continue
            try:
                iface = compiled.flattened_view.top_spec.get_interface(
                    exposed.top_interface
                )
                if not iface.xrt:
                    continue

                # Use runtime group_id if arg_index is available
                if iface.xrt.arg_index is not None and iface.xrt.arg_index in group_ids:
                    bank_map[buffer_id] = group_ids[iface.xrt.arg_index]
                elif iface.xrt.memory_bank_index is not None and group_ids:
                    # Spec index matches a runtime group — use runtime value
                    idx = iface.xrt.memory_bank_index
                    bank_map[buffer_id] = group_ids.get(idx, idx)
                elif iface.xrt.memory_bank_index is not None:
                    bank_map[buffer_id] = iface.xrt.memory_bank_index
            except (KeyError, AttributeError):
                pass

        return bank_map

    def _build_addr_bindings(
        self, compiled: CompiledResult,
    ) -> dict[tuple[int, int], tuple[int, str | None]]:
        """Build address auto_bind substitution map.

        Returns:
            Mapping of (interface_id, reg_offset) → (buffer_id, bits_spec)
            for WRITE_REG commands that contain auto_bind addresses
            needing SHM offset → BO device address translation.
        """
        bindings: dict[tuple[int, int], tuple[int, str | None]] = {}
        view = compiled.flattened_view

        if view is None or view._register_bindings is None:
            return bindings

        # Reverse map: interface_name → interface_id
        name_to_id = {v: k for k, v in compiled.iface_id_to_name.items()}

        for reg_binding in view._register_bindings:
            if reg_binding.auto_bind.value != "address":
                continue

            # Find buffer_id for this tensor via exposed_tensors
            tensor_name = reg_binding.auto_bind.tensor
            # kernel_path format: "top_name.sub_name.iface_name"
            path_parts = reg_binding.kernel_path.split(".")
            sub_kernel_name = path_parts[1] if len(path_parts) > 2 else path_parts[0]
            origin_path = f"{sub_kernel_name}.{tensor_name}"

            buffer_id = None
            for exp_name, exp in view.exposed_tensors.items():
                if exp.origin_path == origin_path:
                    buffer_id = compiled.buffer_ids.get(exp_name)
                    break

            if buffer_id is None:
                continue

            iface_id = name_to_id.get(reg_binding.interface_name)
            if iface_id is None:
                continue

            bindings[(iface_id, reg_binding.absolute_offset)] = (
                buffer_id,
                reg_binding.auto_bind.bits,
            )

        return bindings

    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Execute compiled result on FPGA via XRT.

        1. Initialize device if needed (lazy)
        2. Build interface→IP, buffer→bank, address binding maps
        3. Execute IR commands via CommandInterpreter
        4. Return output buffers in BackendResult
        """
        if self._device is None:
            self._init_device()

        from vten.runtime.interpreter import CommandInterpreter

        ip_map = self._build_ip_map(compiled)
        mem_bank_map = self._build_mem_bank_map(compiled)
        addr_bindings = self._build_addr_bindings(compiled)

        interpreter = CommandInterpreter(
            device=self._device,
            kernel=self._default_ip,
            xrt_module=self._xrt,
            arg_map={},
            poll_timeout_ms=self._poll_timeout_ms,
            ip_map=ip_map,
            mem_bank_map=mem_bank_map,
            addr_bindings=addr_bindings,
        )
        self._interpreter = interpreter

        interpreter.execute(compiled.commands, compiled.tensor_data)

        return BackendResult(
            status=0,
            output_buffers=interpreter.output_buffers,
        )

    @property
    def compile_target(self) -> str:
        """HW backend skips SHM packing (Stage 7)."""
        return "hw"

    def cleanup(self) -> None:
        """Release XRT resources. Idempotent."""
        if self._interpreter is not None:
            self._interpreter.cleanup()
            self._interpreter = None
        self._default_ip = None
        self._group_ids = {}
        self._ips.clear()
        self._device = None
        self._xclbin = None
        self._uuid = None
