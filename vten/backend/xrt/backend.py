"""XrtBackend — XRT/pyxrt-based FPGA backend.

Executes IR Commands on real FPGA hardware via Xilinx Runtime (XRT).
Uses xrt.ip for raw register access (no ap_ctrl_hs assumption).
Uses CommandInterpreter to translate IR Commands to XRT API calls.

Spec reference: 08_backend_abstraction.md §6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vten.backend.base import Backend, BackendResult, CompileTarget, RunContext
from vten.errors import BackendError
from vten.log import format_size

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult


class _ProxyIP:
    """Proxy for an xrt.ip that can't be opened directly.

    Uses another CU's IP handle with offset adjustment to access
    registers at a different base address in the shared PCIe BAR MMIO space.

    This works because xrt::ip::write_register/read_register are raw MMIO
    operations at (base_address + offset), and the CU register spaces share
    the same BAR. By adjusting the offset, we can reach any CU's registers
    through any other CU's IP handle.
    """

    def __init__(self, real_ip: Any, offset_delta: int, name: str = "") -> None:
        self._ip = real_ip
        self._delta = offset_delta
        self._name = name

    def write_register(self, offset: int, value: int) -> None:
        self._ip.write_register(offset + self._delta, value)

    def read_register(self, offset: int) -> int:
        return self._ip.read_register(offset + self._delta)


class XrtBackend(Backend):
    """XRT/pyxrt-based FPGA backend using xrt.ip for raw register access.

    vten.toml [backend.xrt] configuration:
      xclbin_path:      Path to xclbin file
      device_index:     FPGA device index (default: 0)
      kernel_name:      Default IP kernel name within xclbin
      instance_name:    Default IP instance name (optional)
      poll_timeout_ms:  POLL_REG timeout in ms (default: 30000)
    """

    def __init__(self, project_config: dict, persistent: bool = False) -> None:
        from vten.backend.base import RunContext
        self._persistent = persistent
        xrt_cfg = project_config.get("backend", {}).get("xrt", {})
        self._xclbin_path = xrt_cfg.get("xclbin_path", "")
        self._device_index = xrt_cfg.get("device_index", 0)
        self._kernel_name = xrt_cfg.get("kernel_name", "")
        self._instance_name = xrt_cfg.get("instance_name", "")
        target = xrt_cfg.get("target", "hw")
        # hw_emu needs much longer poll timeout (simulation is slow)
        default_timeout = 36000000000 if target == "hw_emu" else 30000000
        self._poll_timeout_ms = xrt_cfg.get("poll_timeout_ms", default_timeout)
        self._config = project_config

        # Default RunContext — callers must call set_run_context() before use.
        self._run_ctx = RunContext()

        # Auto-discover xclbin from kernel build directory
        if not self._xclbin_path:
            self._xclbin_path = self._discover_xclbin()

        # XRT resources (lazy init on first execute)
        self._device: Any = None
        self._xclbin: Any = None
        self._uuid: Any = None
        self._xrt: Any = None  # pyxrt module
        self._default_ip: Any = None
        self._group_ids: dict[int, int] = {}  # arg_index → memory group
        self._ips: dict[str, Any] = {}  # ip_name → xrt.ip
        self._interpreter: Any = None
        self._emu_run_dir: Path | None = None  # hw_emu .run/<PID> to clean up
        self._xrt_ini_created: bool = False  # whether we auto-created xrt.ini
        self._mem_topology: dict[str, int] | None = None  # lazy: bank name → index
        self._cu_base_addresses: dict[str, int] = {}  # CU name → base address

    def set_run_context(self, ctx: RunContext) -> None:
        """Override to re-discover xclbin when RunContext provides kernel_build_dir."""
        super().set_run_context(ctx)
        if not self._xclbin_path:
            self._xclbin_path = self._discover_xclbin()

    def _discover_xclbin(self) -> str:
        """Auto-discover xclbin from kernel build directory.

        Searches kernels/{kernel}/build/xrt/ for .xclbin files.
        For composite kernels, the xclbin is in the composite's build dir.
        """
        import logging

        build_dir = str(self._run_ctx.kernel_build_dir) if self._run_ctx.kernel_build_dir else ""
        if not build_dir:
            return ""

        xrt_dir = Path(build_dir) / "xrt"
        if not xrt_dir.exists():
            return ""

        target = self._config.get("backend", {}).get("xrt", {}).get("target", "hw_emu")
        # Look for target-specific xclbin first, then any xclbin
        xclbins = sorted(xrt_dir.glob(f"*_{target}.xclbin"))
        if not xclbins:
            xclbins = sorted(xrt_dir.glob("*.xclbin"))

        if xclbins:
            logging.getLogger(__name__).info(
                "auto-discovered xclbin: %s", xclbins[0].name
            )
            return str(xclbins[0])
        return ""

    def _setup_emulation_env(self) -> None:
        """Auto-configure emulation environment if xclbin is hw_emu."""
        import os
        import shutil
        import sys

        if not self._xclbin_path:
            return

        xclbin = Path(self._xclbin_path)
        if "_hw_emu" in xclbin.name or "hw_emu" in str(xclbin.parent):
            if not os.environ.get("XCL_EMULATION_MODE"):
                os.environ["XCL_EMULATION_MODE"] = "hw_emu"

            # Ensure emconfig.json is in CWD (XRT requires it there).
            # CWD should already be build/xrt/ (set by cli/run.py),
            # but copy from xclbin dir if needed.
            cwd_emconfig = Path.cwd() / "emconfig.json"
            xclbin_emconfig = xclbin.parent / "emconfig.json"
            if not cwd_emconfig.exists() and xclbin_emconfig.exists():
                shutil.copy2(xclbin_emconfig, cwd_emconfig)

            # Suppress Vitis-EM "Data transfer" and periodic info dumps
            # that pollute the console by mixing with vten's own logging.
            # xrt.ini controls this; create one if user hasn't provided their own.
            self._setup_xrt_ini()

            # Track hw_emu .run/<PID> directory for cleanup.
            # XRT creates this next to sys.executable during emulation.
            exe_dir = Path(sys.executable).resolve().parent
            self._emu_run_dir = exe_dir / ".run" / str(os.getpid())

    def _setup_xrt_ini(self) -> None:
        """Create xrt.ini to suppress verbose hw_emu console output.

        Only writes if no xrt.ini exists in CWD. User-provided xrt.ini
        is never overwritten.

        Settings:
          [Emulation]
          print_infos_in_console=false  — suppresses periodic Data Transfer dumps
          info_suppress=true            — suppresses INFO::[ Vitis-EM ] lines
        """
        import os

        ini_path = Path.cwd() / "xrt.ini"
        if ini_path.exists():
            # User has their own xrt.ini — don't touch it
            return

        ini_content = (
            "[Emulation]\n"
            "print_infos_in_console=false\n"
            "debug_mode=off\n"
            "[Runtime]\n"
            "verbosity=0\n"
        )
        try:
            ini_path.write_text(ini_content)
            self._xrt_ini_created = True
            logger.debug("created xrt.ini to suppress hw_emu console output")
        except OSError as e:
            logger.debug("failed to create xrt.ini: %s", e)

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
                    "Build vten_xrt: cd vten/backend/_xrt_binding && mkdir build "
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

        import time as _time
        import os as _os
        is_emu = _os.environ.get("XCL_EMULATION_MODE") == "hw_emu"
        if is_emu:
            logger.info("starting hw_emu simulator (this may take minutes)...")

        _t0 = _time.monotonic()
        self._device = pyxrt.device(self._device_index)
        self._xclbin = pyxrt.xclbin(self._xclbin_path)
        self._device.load_xclbin(self._xclbin)
        self._uuid = self._xclbin.get_uuid()
        _elapsed = _time.monotonic() - _t0
        logger.info("device ready (%.1fs)", _elapsed)

        # Discover CU names from xclbin (for composite multi-IP routing)
        self._xclbin_cu_names: dict[str, str] = {}  # kernel_name → CU name
        if hasattr(self._xclbin, "get_kernels"):
            import logging
            log = logging.getLogger(__name__)
            for kern in self._xclbin.get_kernels():
                for cu in kern.get_cus():
                    cu_name = cu.get_name()
                    self._xclbin_cu_names[kern.get_name()] = cu_name
            if self._xclbin_cu_names:
                log.debug("xclbin CUs: %s", list(self._xclbin_cu_names.values()))

        # Parse CU base addresses for proxy IP fallback
        self._cu_base_addresses = self._parse_cu_base_addresses()
        if self._cu_base_addresses:
            logger.debug("CU base addresses: %s",
                         {k: f"0x{v:x}" for k, v in self._cu_base_addresses.items()})

        # Cache group_ids for ALL CUs (composite multi-kernel support).
        # Prefer xclbin CONNECTIVITY parsing — it works for all CU types
        # including AP_CTRL_NONE / user_managed IPs without creating
        # xrt::kernel objects (which corrupt HW context on some XRT
        # driver versions, causing BO allocation failures).
        self._per_cu_group_ids: dict[str, dict[int, int]] = {}
        if hasattr(self._xclbin, "get_kernels"):
            self._per_cu_group_ids = self._parse_xclbin_connectivity()
            if self._per_cu_group_ids:
                logger.debug("per-CU group_ids (from xclbin connectivity): %s", {
                    k: dict(v) for k, v in self._per_cu_group_ids.items()
                })

        # Fallback: xrt.kernel group_id() query — only used when xclbin
        # connectivity parsing yields nothing (e.g. older xclbin format).
        # WARNING: this creates temporary xrt::kernel objects which may
        # invalidate HW context on newer XRT drivers with AP_CTRL_NONE CUs.
        if not self._per_cu_group_ids and hasattr(pyxrt, "kernel"):
            for kern_name, cu_name in self._xclbin_cu_names.items():
                try:
                    tmp_kernel = pyxrt.kernel(
                        self._device, self._uuid, kern_name,
                    )
                    gids: dict[int, int] = {}
                    for arg_idx in range(16):
                        try:
                            gids[arg_idx] = tmp_kernel.group_id(arg_idx)
                        except Exception:
                            break
                    self._per_cu_group_ids[kern_name] = gids
                    del tmp_kernel
                except Exception as e:
                    logger.debug(
                        "group_id query failed for CU '%s': %s",
                        cu_name, e,
                    )
            if self._per_cu_group_ids:
                logger.debug("per-CU group_ids: %s", {
                    k: dict(v) for k, v in self._per_cu_group_ids.items()
                })

        # Backward compat: default group_ids from first/default kernel
        if self._kernel_name:
            if self._instance_name:
                ip_name = f"{self._kernel_name}:{{{self._instance_name}}}"
            else:
                ip_name = self._kernel_name
            self._group_ids = self._per_cu_group_ids.get(
                self._kernel_name, {}
            )
            self._default_ip = self._get_or_create_ip(ip_name)

        # INFO-level setup summary
        cu_list = list(self._xclbin_cu_names.values()) or [self._kernel_name or "?"]
        logger.info(
            "XRT: device=%d, xclbin=%s, CUs=%s",
            self._device_index,
            Path(self._xclbin_path).name,
            cu_list,
        )

    def _parse_xclbin_connectivity(self) -> dict[str, dict[int, int]]:
        """Parse xclbin CONNECTIVITY + IP_LAYOUT to get arg→mem_index mapping.

        Fallback for user_managed kernels where xrt.kernel.group_id() fails.
        Tries CONNECTIVITY first, then GROUP_CONNECTIVITY as fallback
        (some xclbins have a corrupted CONNECTIVITY section).

        Returns: {kernel_name: {arg_index: mem_data_index}}.
        """
        import json
        import subprocess
        import tempfile

        xclbin_path = self._xclbin_path

        # Try both section names — GROUP_CONNECTIVITY is a superset that
        # some xclbins use when regular CONNECTIVITY is malformed.
        for section_name, json_key in [
            ("CONNECTIVITY", "connectivity"),
            ("GROUP_CONNECTIVITY", "group_connectivity"),
        ]:
            result: dict[str, dict[int, int]] = {}
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    ip_file = Path(tmpdir) / "ip.json"
                    conn_file = Path(tmpdir) / "conn.json"

                    subprocess.run(
                        ["xclbinutil",
                         "--dump-section", f"IP_LAYOUT:JSON:{ip_file}",
                         "--dump-section", f"{section_name}:JSON:{conn_file}",
                         "--input", str(xclbin_path),
                         "--force"],
                        capture_output=True, check=True,
                    )

                    with open(ip_file) as f:
                        ip_data = json.load(f)
                    with open(conn_file) as f:
                        conn_data = json.load(f)

                # Build IP index → kernel_name mapping
                ip_names: dict[int, str] = {}
                for i, ip in enumerate(ip_data["ip_layout"]["m_ip_data"]):
                    name = ip["m_name"]
                    kern = name.split(":")[0] if ":" in name else name
                    ip_names[i] = kern

                # Build kernel_name → {arg_index: mem_data_index}
                connections = (
                    conn_data.get(json_key, {}).get("m_connection", [])
                    or conn_data.get("connectivity", {}).get("m_connection", [])
                )
                for conn in connections:
                    ip_idx = int(conn["m_ip_layout_index"])
                    arg_idx = int(conn["arg_index"])
                    mem_idx = int(conn["mem_data_index"])
                    kern = ip_names.get(ip_idx)
                    if kern:
                        if kern not in result:
                            result[kern] = {}
                        result[kern][arg_idx] = mem_idx

                if result:
                    logger.debug(
                        "parsed %s: %s",
                        section_name,
                        {k: f"{len(v)} args" for k, v in result.items()},
                    )
                    return result

            except Exception as e:
                logger.debug("Failed to parse %s: %s", section_name, e)

        return {}

    def _parse_cu_base_addresses(self) -> dict[str, int]:
        """Parse IP_LAYOUT from xclbin to get CU base addresses.

        Returns: {"kernel_name:instance_name": base_address, ...}
        E.g.: {"weight_loader:weight_loader_1": 0x802000, ...}
        """
        import json
        import subprocess
        import tempfile

        result: dict[str, int] = {}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ip_file = Path(tmpdir) / "ip.json"
                subprocess.run(
                    ["xclbinutil",
                     "--dump-section", f"IP_LAYOUT:JSON:{ip_file}",
                     "--input", str(self._xclbin_path),
                     "--force"],
                    capture_output=True, check=True,
                )
                with open(ip_file) as f:
                    ip_data = json.load(f)

            for ip in ip_data.get("ip_layout", {}).get("m_ip_data", []):
                name = ip.get("m_name", "").rstrip("\x00").strip()
                base = ip.get("m_base_address", "")
                if name and base:
                    addr = int(base, 16) if isinstance(base, str) else int(base)
                    result[name] = addr

        except FileNotFoundError:
            logger.debug("xclbinutil not found — cannot parse CU base addresses")
        except Exception as e:
            logger.debug("Failed to parse CU base addresses: %s", e)

        return result

    def _get_mem_topology(self) -> dict[str, int]:
        """Get memory bank tag → index mapping (cached, lazy-parsed from xclbin).

        Returns: {"bank0": 0, "bank1": 1, "HBM[0]": 2, "HBM[1]": 3, ...}
        Also creates DDR aliases: "DDR[0]" → index of first DDR bank, etc.
        """
        if self._mem_topology is not None:
            return self._mem_topology
        self._mem_topology = self._parse_mem_topology()
        if self._mem_topology:
            logger.debug("mem_topology: %s", self._mem_topology)
        return self._mem_topology

    def _parse_mem_topology(self) -> dict[str, int]:
        """Parse xclbin MEM_TOPOLOGY to build bank name → index mapping.

        MEM_TOPOLOGY contains the device's physical memory layout.
        Each entry has m_tag (e.g. "bank0", "HBM[0]") and m_type.
        The array index IS the mem_data_index used by XRT BO allocation.
        """
        import json
        import subprocess
        import tempfile

        result: dict[str, int] = {}
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mem_file = Path(tmpdir) / "mem.json"
                subprocess.run(
                    ["xclbinutil",
                     "--dump-section", f"MEM_TOPOLOGY:JSON:{mem_file}",
                     "--input", str(self._xclbin_path),
                     "--force"],
                    capture_output=True, check=True,
                )
                with open(mem_file) as f:
                    mem_data = json.load(f)

            ddr_idx = 0
            for idx, mem in enumerate(
                mem_data.get("mem_topology", {}).get("m_mem_data", [])
            ):
                tag = mem.get("m_tag", "").rstrip("\x00").strip()
                if tag:
                    result[tag] = idx
                # Create "DDR[N]" aliases for DDR banks (tagged "bankN")
                m_type = mem.get("m_type", "")
                if "DDR" in m_type or (
                    tag.startswith("bank") and "HBM" not in m_type
                ):
                    result[f"DDR[{ddr_idx}]"] = idx
                    ddr_idx += 1

        except FileNotFoundError:
            logger.debug("xclbinutil not found — cannot parse MEM_TOPOLOGY")
        except Exception as e:
            logger.debug("Failed to parse MEM_TOPOLOGY: %s", e)

        return result

    def _resolve_memory_bank(self, memory_bank: str) -> int | None:
        """Resolve memory_bank string (e.g. "DDR[0]", "HBM[3]") to bank index.

        Tries exact match first, then DDR[N] → bankN alias.
        """
        import re

        topo = self._get_mem_topology()
        if not topo:
            return None
        # Exact match (handles "HBM[0]", "DDR[0]" alias, etc.)
        if memory_bank in topo:
            return topo[memory_bank]
        # DDR alias fallback: "DDR[N]" → "bankN"
        m = re.match(r"DDR\[(\d+)\]", memory_bank)
        if m:
            bank_tag = f"bank{m.group(1)}"
            if bank_tag in topo:
                return topo[bank_tag]
        return None

    def _get_or_create_ip(self, ip_name: str) -> Any:
        """Get or lazily create an xrt.ip by name.

        Falls back to _ProxyIP if direct xrt.ip creation fails (e.g., CUs
        with many CONNECTIVITY entries that exceed XRT driver limits).
        The proxy uses another CU's IP handle with offset adjustment to
        access registers through the shared PCIe BAR MMIO space.
        """
        if ip_name in self._ips:
            return self._ips[ip_name]

        try:
            self._ips[ip_name] = self._xrt.ip(
                self._device, self._uuid, ip_name,
            )
        except Exception as e:
            logger.warning(
                "xrt.ip creation failed for '%s': %s — attempting proxy IP",
                ip_name, e,
            )
            proxy = self._create_proxy_ip(ip_name)
            if proxy is not None:
                self._ips[ip_name] = proxy
            else:
                raise BackendError(
                    f"Cannot access CU '{ip_name}': {e}. "
                    f"Direct xrt.ip creation failed and no proxy donor available."
                ) from e

        return self._ips[ip_name]

    def _create_proxy_ip(self, ip_name: str) -> _ProxyIP | None:
        """Create a ProxyIP using another CU's IP handle with offset delta.

        Finds the target CU's base address from IP_LAYOUT, picks an existing
        or newly-created donor IP, and creates a proxy that adjusts register
        offsets by (target_base - donor_base).
        """
        if not self._cu_base_addresses:
            return None

        # ip_name format: "kernel:{instance}" — strip braces for IP_LAYOUT lookup
        target_cu = ip_name.replace("{", "").replace("}", "")
        target_base = self._cu_base_addresses.get(target_cu)
        if target_base is None:
            logger.debug("proxy IP: no base address for '%s'", target_cu)
            return None

        # Try existing IPs as donor
        for donor_name, donor_ip in self._ips.items():
            if isinstance(donor_ip, _ProxyIP):
                continue
            donor_cu = donor_name.replace("{", "").replace("}", "")
            donor_base = self._cu_base_addresses.get(donor_cu)
            if donor_base is not None:
                delta = target_base - donor_base
                logger.info(
                    "proxy IP: '%s' (base=0x%x) via '%s' (base=0x%x, delta=0x%x)",
                    ip_name, target_base, donor_name, donor_base, delta,
                )
                return _ProxyIP(donor_ip, delta, ip_name)

        # No existing donor — try to create one from another CU
        for cu_full_name, base in self._cu_base_addresses.items():
            if cu_full_name == target_cu:
                continue
            if ":" in cu_full_name:
                kern, inst = cu_full_name.split(":", 1)
                donor_ip_name = f"{kern}:{{{inst}}}"
            else:
                donor_ip_name = cu_full_name
            try:
                donor_ip = self._xrt.ip(
                    self._device, self._uuid, donor_ip_name,
                )
                self._ips[donor_ip_name] = donor_ip
                delta = target_base - base
                logger.info(
                    "proxy IP: '%s' (base=0x%x) via new '%s' (base=0x%x, delta=0x%x)",
                    ip_name, target_base, donor_ip_name, base, delta,
                )
                return _ProxyIP(donor_ip, delta, ip_name)
            except Exception:
                continue

        return None

    def _build_ip_map(self, compiled: CompiledResult) -> dict[int, Any]:
        """Build interface_id → xrt.ip mapping from compiled result.

        For composite kernels, auto-derives IP names from sub-kernel specs
        and xclbin IP discovery (e.g., wl → weight_loader → weight_loader_1).

        For unit kernels, uses xrt.ip_name from interface config or default_ip.
        """
        import logging
        log = logging.getLogger(__name__)

        ip_map: dict[int, Any] = {}

        if compiled.flattened_view is None:
            return ip_map

        # Build sub-kernel attr→ip_instance mapping for composites
        sub_to_ip = self._build_sub_kernel_ip_map(compiled)

        # Build interface→sub_kernel mapping from interface_mappings
        iface_to_sub: dict[str, str] = {}
        for m in compiled.flattened_view.interface_mappings:
            if m.top_interface:
                iface_to_sub[m.top_interface] = m.sub_kernel

        for iface_id, iface_name in compiled.iface_id_to_name.items():
            try:
                iface_spec = compiled.flattened_view.top_spec.get_interface(
                    iface_name
                )
            except (KeyError, AttributeError):
                if self._default_ip is not None:
                    ip_map[iface_id] = self._default_ip
                continue

            # Priority: explicit xrt.ip_name > composite auto-routing > default_ip
            if iface_spec.xrt and iface_spec.xrt.ip_name:
                ip_map[iface_id] = self._get_or_create_ip(
                    iface_spec.xrt.ip_name
                )
            elif iface_name in iface_to_sub and iface_to_sub[iface_name] in sub_to_ip:
                ip_map[iface_id] = self._get_or_create_ip(
                    sub_to_ip[iface_to_sub[iface_name]]
                )
            elif self._default_ip is not None:
                ip_map[iface_id] = self._default_ip

        if sub_to_ip:
            log.debug("composite IP routing: %s", sub_to_ip)

        return ip_map

    def _build_sub_kernel_ip_map(self, compiled: CompiledResult) -> dict[str, str]:
        """Map sub-kernel attr names to xclbin CU names.

        Uses xclbin introspection (get_kernels/get_cus) for exact CU names.
        Falls back to v++ convention "{kernel_name}:{kernel_name}_1".

        E.g., {"wl": "weight_loader:weight_loader_1", "fmap": "fmapIO:fmapIO_1"}
        """
        view = compiled.flattened_view
        if view is None or not view.sub_kernels:
            return {}

        cu_names = getattr(self, "_xclbin_cu_names", {})

        sub_to_ip: dict[str, str] = {}
        for attr_name, ki in view.sub_kernels.items():
            kernel_name = ki.spec.kernel_name
            if kernel_name in cu_names:
                # CU name from xclbin is "kern:inst", xrt.ip needs "kern:{inst}"
                cu_name = cu_names[kernel_name]
                if ":" in cu_name:
                    kern, inst = cu_name.split(":", 1)
                    sub_to_ip[attr_name] = f"{kern}:{{{inst}}}"
                else:
                    sub_to_ip[attr_name] = cu_name
            else:
                # Fallback: v++ single-instance convention
                sub_to_ip[attr_name] = f"{kernel_name}:{{{kernel_name}_1}}"

        return sub_to_ip

    def _build_mem_bank_map(
        self, compiled: CompiledResult,
    ) -> dict[int, int]:
        """Build buffer_id → memory bank index mapping.

        For composite kernels, uses per-CU group_ids from _init_device.
        For each exposed tensor, finds the owning sub-kernel's CU to
        query the correct group_id(arg_index).
        """
        bank_map: dict[int, int] = {}

        if compiled.flattened_view is None:
            return bank_map

        view = compiled.flattened_view
        per_cu_gids = getattr(self, "_per_cu_group_ids", {})
        default_gids: dict[int, int] = getattr(self, "_group_ids", {})

        # Build interface → sub_kernel_name mapping
        iface_to_sub: dict[str, str] = {}
        for m in view.interface_mappings:
            if m.top_interface:
                iface_to_sub[m.top_interface] = m.sub_kernel

        for name, exposed in view.exposed_tensors.items():
            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
                if not iface.xrt:
                    continue

                # For composite kernels, find the owning sub-kernel's CU
                sub_name = iface_to_sub.get(exposed.top_interface)
                group_ids = default_gids
                if sub_name and view.sub_kernels:
                    ki = view.sub_kernels.get(sub_name)
                    if ki and ki.spec:
                        kern_name = ki.spec.kernel_name
                        group_ids = per_cu_gids.get(kern_name, default_gids)

                # Map logical buffer_id
                # Priority: group_id(arg) > memory_bank_index > memory_bank string
                buffer_id = compiled.buffer_ids.get(name)
                if buffer_id is not None:
                    if iface.xrt.arg_index is not None and iface.xrt.arg_index in group_ids:
                        bank_map[buffer_id] = group_ids[iface.xrt.arg_index]
                    elif iface.xrt.memory_bank_index is not None:
                        bank_map[buffer_id] = iface.xrt.memory_bank_index
                    elif iface.xrt.memory_bank is not None:
                        resolved = self._resolve_memory_bank(iface.xrt.memory_bank)
                        if resolved is not None:
                            bank_map[buffer_id] = resolved

                # Map per-port buffer_ids (array/split interfaces)
                if exposed._port_buffers:
                    base_arg = iface.xrt.arg_index or 0
                    for port_idx, port_name in enumerate(exposed._port_buffers):
                        port_bid = compiled.buffer_ids.get(f"{name}:{port_name}")
                        if port_bid is None:
                            continue
                        # Array elements: each port may have its own arg_index
                        arg_idx = base_arg + port_idx
                        if arg_idx in group_ids:
                            bank_map[port_bid] = group_ids[arg_idx]
                        elif group_ids and base_arg in group_ids:
                            # Extrapolate: if base arg maps to group N,
                            # port i maps to group N + i (e.g. HBM array)
                            bank_map[port_bid] = group_ids[base_arg] + port_idx
                        elif iface.xrt.memory_bank_index is not None:
                            bank_map[port_bid] = iface.xrt.memory_bank_index
                        elif iface.xrt.memory_bank is not None:
                            # Resolve template: "HBM[{i}]" → "HBM[0]", "HBM[1]", ...
                            bank_name = iface.xrt.memory_bank.replace("{i}", str(port_idx))
                            resolved = self._resolve_memory_bank(bank_name)
                            if resolved is not None:
                                bank_map[port_bid] = resolved

            except (KeyError, AttributeError):
                pass

        return bank_map

    def _build_addr_bindings(
        self, compiled: CompiledResult,
    ) -> dict[tuple[int, int], tuple[int, str | None, int]]:
        """Build address auto_bind substitution map.

        Returns:
            Mapping of (interface_id, reg_offset) → (buffer_id, bits_spec, byte_offset)
            for WRITE_REG commands that contain auto_bind addresses
            needing SHM offset → BO device address translation.
        """
        bindings: dict[tuple[int, int], tuple[int, str | None, int]] = {}
        # Per-port expansions override main-loop entries (per-port BO
        # addresses must win over single-tensor auto_bind with byte offsets).
        port_overrides: dict[tuple[int, int], tuple[int, str | None, int]] = {}
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

            # Resolve byte offset from auto_bind.offset (may be expression)
            byte_offset = 0
            if reg_binding.auto_bind.offset is not None:
                if isinstance(reg_binding.auto_bind.offset, int):
                    byte_offset = reg_binding.auto_bind.offset
                else:
                    # Expression — already resolved during compile
                    # Use resolved_value minus base address to extract offset
                    sub = view.sub_kernels.get(sub_kernel_name)
                    if sub and sub._resolver:
                        byte_offset = int(
                            sub._resolver.resolve(reg_binding.auto_bind.offset)
                        )

            bindings[(iface_id, reg_binding.absolute_offset)] = (
                buffer_id,
                reg_binding.auto_bind.bits,
                byte_offset,
            )

            # For array tensors: expand auto_bind to per-port registers.
            # Register naming convention: "addr_lo_0" → "addr_lo_1".."_N"
            # at 8-byte stride from the base (lo/hi pair = 8 bytes per port).
            reg_name = reg_binding.register_name
            # Extract the raw reg name (after sub_kernel prefix "wl.")
            raw_name = reg_name.split(".")[-1] if "." in reg_name else reg_name
            if raw_name.endswith("_0") and "lo" in raw_name:
                for exp_name, exp in view.exposed_tensors.items():
                    if exp.origin_path == origin_path and exp._port_buffers:
                        port_names = list(exp._port_buffers.keys())
                        base_offset = reg_binding.absolute_offset
                        for port_idx in range(1, len(port_names)):
                            port_bid = compiled.buffer_ids.get(
                                f"{exp_name}:{port_names[port_idx]}"
                            )
                            if port_bid is None:
                                continue
                            # lo register: base + 8*port_idx
                            lo_off = base_offset + 8 * port_idx
                            # hi register: base + 8*port_idx + 4
                            hi_off = base_offset + 8 * port_idx + 4
                            port_overrides[(iface_id, lo_off)] = (
                                port_bid, "31:0", 0,
                            )
                            port_overrides[(iface_id, hi_off)] = (
                                port_bid, "63:32", 0,
                            )
                        break

        # Per-port overrides take precedence: each port's BO has its own
        # device address starting at offset 0, not a shared base + stride.
        bindings.update(port_overrides)
        return bindings

    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Execute compiled result on FPGA via XRT.

        1. Initialize device if needed (lazy)
        2. Build interface→IP, buffer→bank, address binding maps
        3. Execute IR commands via CommandInterpreter
        4. Return output buffers in BackendResult

        Persistent mode: reuses CommandInterpreter (and its BO pool)
        across calls. Prebound BOs from compiled.prebound_buffers are
        injected into the interpreter before execution.
        """
        if self._device is None:
            logger.info("initializing XRT device (first execution)...")
            self._init_device()

        from vten.backend.xrt.interpreter import CommandInterpreter

        # Diagnostic: dump flattened_view structure for debugging
        if compiled.flattened_view is not None:
            view = compiled.flattened_view
            logger.debug(
                "flattened_view: sub_kernels=%s, exposed=%s, iface_map=%s",
                {k: v.spec.kernel_name for k, v in view.sub_kernels.items()},
                list(view.exposed_tensors.keys()),
                [(m.sub_kernel, m.sub_interface, m.top_interface)
                 for m in view.interface_mappings],
            )
            logger.debug("iface_id_to_name: %s", compiled.iface_id_to_name)
            logger.debug("buffer_ids: %s", compiled.buffer_ids)

        ip_map = self._build_ip_map(compiled)
        mem_bank_map = self._build_mem_bank_map(compiled)
        addr_bindings = self._build_addr_bindings(compiled)
        logger.debug("ip_map: %s", {k: getattr(v, '_name', str(v)) for k, v in ip_map.items()})

        quiet = compiled.mode == "inference"
        if self._persistent and self._interpreter is not None:
            interpreter = self._interpreter
            interpreter.update_maps(ip_map, mem_bank_map, addr_bindings)
            interpreter._quiet = quiet
        else:
            interpreter = CommandInterpreter(
                device=self._device,
                kernel=self._default_ip,
                xrt_module=self._xrt,
                arg_map={},
                poll_timeout_ms=self._poll_timeout_ms,
                ip_map=ip_map,
                mem_bank_map=mem_bank_map,
                addr_bindings=addr_bindings,
                quiet=quiet,
            )
            self._interpreter = interpreter

        # Inject prebound device buffers (inference: BO already on device)
        if compiled.prebound_buffers:
            for buffer_id, bo in compiled.prebound_buffers.items():
                interpreter._buffers[buffer_id] = bo
                interpreter._prebound.add(buffer_id)
                # Record bank so _preallocate_bound_bos won't replace with wrong bank
                if buffer_id in mem_bank_map:
                    interpreter._buffer_banks[buffer_id] = mem_bank_map[buffer_id]
            logger.debug(
                "prebound injected: %s (prebound_set=%s)",
                list(compiled.prebound_buffers.keys()),
                interpreter._prebound,
            )

        logger.debug("mem_bank_map: %s", mem_bank_map)
        logger.debug("addr_bindings: %s", addr_bindings)
        total_data = sum(len(v) for v in compiled.tensor_data.values())
        _log = logger.debug if quiet else logger.info
        _log(
            "executing %d commands (%d tensors, %s)",
            len(compiled.commands), len(compiled.tensor_data),
            format_size(total_data),
        )
        interpreter.execute(compiled.commands, compiled.tensor_data)

        return BackendResult(
            status=0,
            output_buffers=interpreter.output_buffers,
        )

    def get_execution_summary(self) -> object | None:
        """Return the last ExecutionSummary from the interpreter, or None."""
        if self._interpreter is not None:
            return self._interpreter._summary
        return None

    def get_buffer_object(self, buffer_id: int) -> object | None:
        """Return XRT BO for buffer_id from interpreter's buffer pool."""
        if self._interpreter is not None:
            return self._interpreter._buffers.get(buffer_id)
        return None

    def inject_prebound(self, buffer_id: int, bo: object) -> None:
        """Inject a pre-existing BO into interpreter's buffer pool."""
        if self._interpreter is not None:
            self._interpreter._buffers[buffer_id] = bo
            self._interpreter._prebound.add(buffer_id)

    def working_directory(self, kernel_dir: Path, project_dir: Path) -> Path:
        """XRT dumps artifacts (emconfig, traces) to CWD — isolate them."""
        xrt_dir = kernel_dir / "build" / "xrt"
        xrt_dir.mkdir(parents=True, exist_ok=True)
        return xrt_dir

    @property
    def compile_target(self) -> CompileTarget:
        return CompileTarget.HW

    def cleanup(self) -> None:
        """Release XRT resources and remove hw_emu artifacts. Idempotent."""
        if self._interpreter is not None:
            self._interpreter.cleanup()
            self._interpreter = None
        self._default_ip = None
        self._group_ids = {}
        self._ips.clear()
        self._device = None
        self._xclbin = None
        self._uuid = None
        # Clean up auto-created xrt.ini
        if self._xrt_ini_created:
            try:
                (Path.cwd() / "xrt.ini").unlink(missing_ok=True)
            except OSError:
                pass
            self._xrt_ini_created = False
        self._cleanup_emu_run_dir()

    def _cleanup_emu_run_dir(self) -> None:
        """Remove hw_emu .run/<PID> directories.

        XRT hw_emu creates <sys.executable>/../.run/<PID>/ containing
        unzipped xclbin, BO state files, etc. These are not cleaned up
        automatically and can accumulate to many GB over time.

        Cleans up both the current PID's directory and any orphaned
        directories from previous runs whose PIDs are no longer alive.
        """
        if self._emu_run_dir is None:
            return

        run_dir = self._emu_run_dir
        self._emu_run_dir = None
        run_parent = run_dir.parent  # .run/

        import shutil

        # Clean current PID's directory
        if run_dir.is_dir():
            try:
                shutil.rmtree(run_dir)
                logger.debug("cleaned up hw_emu artifacts: %s", run_dir)
            except OSError as e:
                logger.warning("failed to clean up %s: %s", run_dir, e)

        # Clean orphaned .run/<PID> directories from crashed previous runs
        if run_parent.is_dir():
            import os
            import signal

            for entry in run_parent.iterdir():
                if not entry.is_dir() or not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                # Check if PID is still alive
                try:
                    os.kill(pid, signal.SIG_DFL)
                    # PID alive — skip
                except ProcessLookupError:
                    # PID dead — orphaned, safe to remove
                    try:
                        shutil.rmtree(entry)
                        logger.debug("cleaned up orphaned hw_emu dir: %s", entry)
                    except OSError:
                        pass
                except PermissionError:
                    # Not our process — skip
                    pass
