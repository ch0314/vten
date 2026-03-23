"""vten build: compilation pipeline.

Spec reference: 06_codegen_and_cli.md §4.3
"""

from __future__ import annotations

import struct
from pathlib import Path

from vten.cli.config import load_project_config
from vten.codegen.sv_generator import SVGenerator
from vten.runtime.ir import BFMConfig
from vten.runtime.shm import CONTROL_SIZE, SHM_MAGIC, PROTOCOL_VERSION
from vten.spec.models import InterfaceSpec, KernelSpec, Protocol
from vten.spec.parser import parse_kernel_spec


def _infer_bfm_role(iface: InterfaceSpec) -> str:
    """Infer BFM role from protocol and interface conventions.

    AXI4-Stream: rtl_port 's_*' → DUT is slave → BFM is master (pushes data)
                 rtl_port 'm_*' → DUT is master → BFM is slave (pulls data)
    AXI4:        BFM is always slave (DUT initiates reads/writes)
    AXI4-Lite:   BFM is always master (drives register access)
    """
    if iface.protocol == Protocol.AXI4L:
        return "master"
    if iface.protocol == Protocol.AXI4:
        return "slave"
    # AXI4-Stream: infer from rtl_port prefix
    if iface.rtl_port and iface.rtl_port.startswith("s_"):
        return "master"  # DUT slave input → BFM drives data
    return "slave"  # DUT master output → BFM receives data


def _derive_bfm_configs(spec: KernelSpec) -> list[BFMConfig]:
    """Derive BFMConfig list from KernelSpec interfaces."""
    configs: list[BFMConfig] = []
    for name, iface in spec.interfaces.items():
        cfg = BFMConfig(
            interface_name=name,
            protocol=iface.protocol,
            data_width=iface.data_width or 256,
            addr_width=iface.addr_width or 64,
            role=_infer_bfm_role(iface),
        )
        configs.append(cfg)
    return configs


def build_project(project_dir: str, config_overrides: dict | None = None) -> None:
    """Build project: parse config → parse spec → codegen → scripts."""
    project = Path(project_dir)
    config = load_project_config(project)

    if config_overrides:
        params = config.setdefault("parameters", {})
        params.update(config_overrides)

    build_dir = project / "build"
    for subdir in ["generated", "scripts", "shm", "lib"]:
        (build_dir / subdir).mkdir(parents=True, exist_ok=True)

    if "rtl" not in config:
        raise ValueError("Missing [rtl] section in vten.toml")

    rtl_cfg = config["rtl"]
    top_module = rtl_cfg.get("top_module", "tb_top")

    # Try to parse kernel_spec.yaml from specs/ directory
    spec: KernelSpec | None = None
    specs_dir = project / "specs"
    if specs_dir.exists():
        yaml_files = list(specs_dir.glob("*.yaml")) + list(specs_dir.glob("*.yml"))
        if yaml_files:
            try:
                spec = parse_kernel_spec(yaml_files[0])
            except Exception:
                pass

    # Fall back to minimal spec if no yaml found or parsing failed
    if spec is None:
        spec = KernelSpec(
            kernel_name=top_module,
            rtl_top=rtl_cfg.get("sources", [""])[0] if rtl_cfg.get("sources") else "",
            interfaces={},
        )

    bfm_configs = _derive_bfm_configs(spec)

    # Extract num_commands from pre-built SHM image if available
    num_commands = 0
    shm_bin = build_dir / "shm" / "kernel_task.bin"
    if shm_bin.exists():
        shm_data = shm_bin.read_bytes()
        if len(shm_data) >= 0x14:
            num_commands = struct.unpack_from("<I", shm_data, 0x10)[0]

    gen = SVGenerator(
        kernel_spec=spec,
        bfm_configs=bfm_configs,
        project_config=config,
    )

    gen.generate(str(build_dir / "generated"), num_commands=num_commands)

    # Move scripts to scripts/
    gen_dir = build_dir / "generated"
    scripts_dir = build_dir / "scripts"
    for script in ["build.tcl", "run.tcl"]:
        src = gen_dir / script
        dst = scripts_dir / script
        if src.exists():
            dst.write_text(src.read_text())
            src.unlink()

    # Write minimal SHM image placeholder
    shm_image = bytearray(CONTROL_SIZE)
    struct.pack_into("<I", shm_image, 0, SHM_MAGIC)
    struct.pack_into("<I", shm_image, 4, PROTOCOL_VERSION)
    (build_dir / "shm" / "kernel_task.bin").write_bytes(bytes(shm_image))
