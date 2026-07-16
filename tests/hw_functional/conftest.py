"""Hardware functional test fixtures.

Session-scoped compilation of verilator modules + test-scoped simulator instances.
These tests exercise the generated BFMs, the SHM controller, and the command
scheduler against a Verilator-compiled model of the vTen SystemVerilog.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

# ── Paths ──

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VTEN_SV = PROJECT_ROOT / "vten" / "sv"
STUBS_DIR = PROJECT_ROOT / "tests" / "hw_functional" / "stubs"
WRAPPERS_DIR = PROJECT_ROOT / "tests" / "hw_functional" / "wrappers"
BUILD_DIR = PROJECT_ROOT / "build" / "verilator"


def _find_verilator() -> str | None:
    return shutil.which("verilator")


def _find_verilator_include() -> Path | None:
    """Find verilator include directory."""
    verilator = _find_verilator()
    if not verilator:
        return None
    try:
        result = subprocess.run(
            [verilator, "--getenv", "VERILATOR_ROOT"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            root = Path(result.stdout.strip())
            inc = root / "include"
            if inc.exists():
                return inc
    except Exception:
        pass
    # Fallback
    for p in [Path("/usr/local/share/verilator/include"),
              Path("/usr/share/verilator/include")]:
        if p.exists():
            return p
    return None


VERILATOR_BIN = _find_verilator()
VERILATOR_INC = _find_verilator_include()

requires_verilator = pytest.mark.skipif(
    VERILATOR_BIN is None or VERILATOR_INC is None,
    reason="verilator not found",
)


# ── SHM Constants (imported from single source of truth) ──

from vten.backend.sim.shm_constants import (  # noqa: E402
    CACHE_LINE,
    BUF_DESC_SIZE,
    CMD_SLOT_SIZE,
    CONTROL_SIZE,
    SHM_MAGIC,
    PROTOCOL_VERSION,
    STATS_SLOT_SIZE,
)


# ── Compilation helpers ──

def _verilate(
    module_sv: str,
    top_module: str,
    build_name: str,
    *,
    sv_dir: Path | None = None,
) -> Path:
    """Run verilator --cc on a SystemVerilog file. Returns Mdir path.

    If sv_dir is given, module_sv is resolved relative to that directory
    (useful for wrapper modules in tests/hw_functional/wrappers/).
    """
    mdir = BUILD_DIR / build_name
    mdir.mkdir(parents=True, exist_ok=True)

    base_dir = sv_dir if sv_dir is not None else VTEN_SV
    cmd = [
        VERILATOR_BIN, "--cc", str(base_dir / module_sv),
        "--top-module", top_module,
        "-I" + str(VTEN_SV),
        "-Mdir", str(mdir),
        "--prefix", f"V{top_module}",
        "-Wno-WIDTHEXPAND", "-Wno-CASEINCOMPLETE",
        "-Wno-IGNOREDRETURN", "-Wno-WIDTHTRUNC",
        "-Wno-DEPRECATED", "-Wno-WIDTHCONCAT",
        "-Wno-MULTIDRIVEN", "-Wno-SIDEEFFECT",
        # Scheduler/controller MAX_CMDS loops must fully unroll so NBA array
        # writes get constant indices (avoids BLKLOOPINIT miscompilation).
        # Same defaults as vten/build/verilator_build.py _stage_verilate().
        "--unroll-count", "256",
        "--unroll-stmts", "200000",
        "--threads", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"verilator failed: {result.stderr}")
    return mdir


def _compile_mock(mdir: Path) -> Path:
    """Compile dpi_mock.c into an object file."""
    obj = mdir / "dpi_mock.o"
    cmd = [
        "gcc", "-c", "-fPIC",
        "-I" + str(VERILATOR_INC),
        "-I" + str(VERILATOR_INC / "vltstd"),
        "-I" + str(mdir),
        "-I" + str(STUBS_DIR),
        "-o", str(obj),
        str(STUBS_DIR / "dpi_mock.c"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"gcc mock compile failed: {result.stderr}")
    return obj


def _build_driver(
    mdir: Path,
    driver_cpp: str,
    output_name: str,
    extra_cpps: list[str] | None = None,
) -> Path:
    """Build a verilator driver executable."""
    # Collect all generated .cpp files
    gen_cpps = sorted(mdir.glob("V*.cpp"))
    mock_obj = _compile_mock(mdir)

    exe = mdir / output_name
    cmd = [
        "g++", "-std=c++17", "-O2", "-w",
        "-o", str(exe),
        "-I" + str(VERILATOR_INC),
        "-I" + str(VERILATOR_INC / "vltstd"),
        "-I" + str(mdir),
        "-I" + str(STUBS_DIR),
        str(WRAPPERS_DIR / driver_cpp),
    ]
    cmd += [str(f) for f in gen_cpps]
    cmd += [
        str(VERILATOR_INC / "verilated.cpp"),
        str(VERILATOR_INC / "verilated_dpi.cpp"),
        str(VERILATOR_INC / "verilated_threads.cpp"),
        str(mock_obj),
        "-lpthread",
    ]
    if extra_cpps:
        cmd += extra_cpps

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"g++ driver build failed: {result.stderr}")
    return exe


# ── Session-scoped fixtures (compile once) ──

@pytest.fixture(scope="session")
def shm_ctrl_driver() -> Path:
    """Compile vten_shm_controller driver. Returns executable path."""
    if VERILATOR_BIN is None or VERILATOR_INC is None:
        pytest.skip("verilator not found")
    mdir = _verilate(
        "vten_shm_controller.sv", "vten_shm_controller", "shm_ctrl"
    )
    return _build_driver(mdir, "shm_ctrl_driver.cpp", "shm_ctrl_driver")


# ── Session-scoped: axilite BFM ──

@pytest.fixture(scope="session")
def axilite_driver() -> Path:
    """Compile vten_bfm_axilite driver. Returns executable path."""
    if VERILATOR_BIN is None or VERILATOR_INC is None:
        pytest.skip("verilator not found")
    mdir = _verilate(
        "tb_bfm_axilite.sv", "tb_bfm_axilite", "axilite",
        sv_dir=WRAPPERS_DIR,
    )
    return _build_driver(mdir, "axilite_driver.cpp", "axilite_driver")


# ── Session-scoped: axi4s BFM ──

@pytest.fixture(scope="session")
def axi4s_driver() -> Path:
    """Compile vten_bfm_axi4s driver (MASTER mode). Returns executable path."""
    if VERILATOR_BIN is None or VERILATOR_INC is None:
        pytest.skip("verilator not found")
    mdir = _verilate(
        "tb_bfm_axi4s.sv", "tb_bfm_axi4s", "axi4s",
        sv_dir=WRAPPERS_DIR,
    )
    return _build_driver(mdir, "axi4s_driver.cpp", "axi4s_driver")


# ── Session-scoped: axi4 BFM ──

@pytest.fixture(scope="session")
def axi4_driver() -> Path:
    """Compile vten_bfm_axi4 driver. Returns executable path."""
    if VERILATOR_BIN is None or VERILATOR_INC is None:
        pytest.skip("verilator not found")
    mdir = _verilate(
        "tb_bfm_axi4.sv", "tb_bfm_axi4", "axi4",
        sv_dir=WRAPPERS_DIR,
    )
    return _build_driver(mdir, "axi4_driver.cpp", "axi4_driver")


# ── Session-scoped: scheduler ──

@pytest.fixture(scope="session")
def scheduler_driver() -> Path:
    """Compile vten_command_scheduler driver. Returns executable path."""
    if VERILATOR_BIN is None or VERILATOR_INC is None:
        pytest.skip("verilator not found")
    mdir = _verilate(
        "tb_scheduler.sv", "tb_scheduler", "scheduler",
        sv_dir=WRAPPERS_DIR,
    )
    return _build_driver(mdir, "scheduler_driver.cpp", "scheduler_driver")


# ── Test-scoped fixtures ──

@pytest.fixture
def shm_ctrl_sim(shm_ctrl_driver):
    """Create a VerilatorSim instance for shm_controller."""
    from tests.hw_functional.sim_harness import VerilatorSim

    with VerilatorSim(shm_ctrl_driver) as sim:
        yield sim


@pytest.fixture
def axilite_sim(axilite_driver):
    """Create a VerilatorSim instance for axilite BFM."""
    from tests.hw_functional.sim_harness import VerilatorSim

    with VerilatorSim(axilite_driver) as sim:
        yield sim


@pytest.fixture
def axi4s_sim(axi4s_driver):
    """Create a VerilatorSim instance for axi4s BFM (MASTER mode)."""
    from tests.hw_functional.sim_harness import VerilatorSim

    with VerilatorSim(axi4s_driver) as sim:
        yield sim


@pytest.fixture
def axi4_sim(axi4_driver):
    """Create a VerilatorSim instance for axi4 BFM."""
    from tests.hw_functional.sim_harness import VerilatorSim

    with VerilatorSim(axi4_driver) as sim:
        yield sim


@pytest.fixture
def scheduler_sim(scheduler_driver):
    """Create a VerilatorSim instance for scheduler."""
    from tests.hw_functional.sim_harness import VerilatorSim

    with VerilatorSim(scheduler_driver) as sim:
        yield sim


# ── SHM image builders ──

def build_shm_image(
    num_commands: int,
    commands: list[dict] | None = None,
    num_buffers: int = 0,
    buffer_descs: list[dict] | None = None,
    host_status: int = 1,  # CMD_READY
    flags: int = 0x01,  # STATS_ENABLED
    timeout_ms: int = 0,
) -> bytearray:
    """Build a complete SHM image for testing.

    Each command dict should have: opcode, cmd_id, interface_id, protocol,
    role, buffer_id, size, and optionally phys_addr, reg_offset, reg_value,
    reg_mask, reg_expected, golden_buf, probe, sync, deps, commit_deps.
    """
    cmd_off = CONTROL_SIZE
    stats_off = cmd_off + CMD_SLOT_SIZE * num_commands
    buf_desc_off = stats_off + STATS_SLOT_SIZE * num_commands
    data_off = buf_desc_off + BUF_DESC_SIZE * num_buffers
    # Align data_off
    data_off = (data_off + CACHE_LINE - 1) & ~(CACHE_LINE - 1)
    total_size = data_off  # No actual data for now

    image = bytearray(total_size)

    # Control header
    struct.pack_into("<I", image, 0x00, SHM_MAGIC)
    struct.pack_into("<I", image, 0x04, PROTOCOL_VERSION)
    struct.pack_into("<I", image, 0x08, host_status)
    struct.pack_into("<I", image, 0x0C, 0)  # backend_status = IDLE
    struct.pack_into("<I", image, 0x10, num_commands)
    struct.pack_into("<I", image, 0x14, num_buffers)
    struct.pack_into("<Q", image, 0x18, cmd_off)
    struct.pack_into("<Q", image, 0x20, stats_off)
    struct.pack_into("<Q", image, 0x28, buf_desc_off)
    struct.pack_into("<Q", image, 0x30, data_off)
    struct.pack_into("<Q", image, 0x38, total_size)
    struct.pack_into("<I", image, 0x88, flags)
    struct.pack_into("<I", image, 0x8C, timeout_ms)

    # Commands
    if commands:
        for cmd in commands:
            idx = cmd.get("cmd_id", 0)
            off = cmd_off + idx * CMD_SLOT_SIZE
            struct.pack_into("<H", image, off + 0x00, cmd.get("opcode", 2))
            struct.pack_into("<H", image, off + 0x02, idx)
            struct.pack_into("<H", image, off + 0x04, cmd.get("interface_id", 0))
            struct.pack_into("<B", image, off + 0x06, cmd.get("protocol", 1))
            struct.pack_into("<B", image, off + 0x07, cmd.get("role", 0))
            struct.pack_into("<H", image, off + 0x08, cmd.get("buffer_id", 0))
            probe = 1 if cmd.get("probe", False) else 0
            struct.pack_into("<B", image, off + 0x0A, probe)
            sync_flag = 1 if cmd.get("sync", False) else 0
            struct.pack_into("<B", image, off + 0x0B, sync_flag)
            struct.pack_into("<I", image, off + 0x0C, cmd.get("size", 256))
            struct.pack_into("<Q", image, off + 0x10, cmd.get("phys_addr", 0))
            struct.pack_into("<I", image, off + 0x18, cmd.get("reg_offset", 0))
            struct.pack_into("<I", image, off + 0x1C, cmd.get("reg_value", 0))
            struct.pack_into("<I", image, off + 0x20, cmd.get("reg_mask", 0))
            struct.pack_into("<I", image, off + 0x24, cmd.get("reg_expected", 0))
            struct.pack_into("<H", image, off + 0x28, cmd.get("golden_buf", 0))

            deps = cmd.get("deps", [])
            commit_deps = cmd.get("commit_deps", [])
            struct.pack_into("<B", image, off + 0x2A, len(deps))
            struct.pack_into("<B", image, off + 0x2B, len(commit_deps))
            for j in range(4):
                d = deps[j] if j < len(deps) else 0xFFFF
                struct.pack_into("<H", image, off + 0x2C + j * 2, d)
            for j in range(4):
                d = commit_deps[j] if j < len(commit_deps) else 0xFFFF
                struct.pack_into("<H", image, off + 0x34 + j * 2, d)

    return image


# ── OpCode constants ──

OP_LOAD = 1
OP_PUSH = 2
OP_PULL = 3
OP_STORE = 4
OP_WRITE_REG = 5
OP_READ_REG = 6
OP_POLL_REG = 7
OP_BARRIER = 8

# ── Protocol constants ──
PROTO_AXI4S = 1
PROTO_AXI4 = 2
PROTO_AXI4L = 3

# ── Role constants ──
ROLE_MASTER = 0
ROLE_SLAVE = 1

# ── Host/Backend status ──
HOST_IDLE = 0
HOST_CMD_READY = 1
HOST_ACK = 2
HOST_SHUTDOWN = 3
BACKEND_IDLE = 0
BACKEND_RUNNING = 1
BACKEND_DONE = 2
BACKEND_ERROR = 3

# ── FSM states ──
S_INIT = 0
S_WAIT_HOST = 1
S_LOAD_BATCH = 2
S_FEED = 3
S_EXECUTE = 4
S_DRAIN = 5
S_COMPLETE = 6
S_ERROR = 7
S_SHUTDOWN = 8
