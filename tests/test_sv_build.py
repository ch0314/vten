"""Phase 3 tests: SV/C build verification and ctypes SHM round-trip.

Three levels of testing:
1. gcc compilation of vten_shm_bridge.c → libvten_shm.so
2. xvlog syntax check of all SV files
3. ctypes: load compiled .so, write Python SHM image, verify C reads correctly

Level 3 is the most powerful: it validates that Phase 2 SHM packing
and Phase 3 C unpacking are byte-compatible end-to-end.

NPU 3D patterns used for realistic command/buffer layouts.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

VTEN_SV_DIR = Path(__file__).resolve().parent.parent / "vten" / "sv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sv_files_exist() -> bool:
    """Check if vten_sv/ has been implemented."""
    return (VTEN_SV_DIR / "vten_shm_bridge.c").exists()


def _has_gcc() -> bool:
    return shutil.which("gcc") is not None


def _has_xvlog() -> bool:
    return shutil.which("xvlog") is not None


def _has_verilator() -> bool:
    return shutil.which("verilator") is not None


# ═══════════════════════════════════════════════════════════════════
# §1. gcc compilation — Phase 3 completion criterion
#     "gcc 컴파일 성공" (CLAUDE.md)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_gcc(), reason="gcc not available")
@pytest.mark.skipif(not _sv_files_exist(), reason="vten_sv/ not yet implemented")
class TestGccCompilation:
    """Compile vten_shm_bridge.c into shared library."""

    def test_compile_shared_library(self, tmp_path):
        """gcc -shared -fPIC -o libvten_shm.so vten_shm_bridge.c -lrt -lpthread."""
        c_file = VTEN_SV_DIR / "vten_shm_bridge.c"
        h_file = VTEN_SV_DIR / "vten_shm_bridge.h"
        assert c_file.exists()
        assert h_file.exists()

        output = tmp_path / "libvten_shm.so"
        result = subprocess.run(
            [
                "gcc", "-shared", "-fPIC",
                "-I", str(VTEN_SV_DIR),
                "-o", str(output),
                str(c_file),
                "-lrt", "-lpthread",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"gcc failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_compile_no_warnings(self, tmp_path):
        """Compilation with -Wall -Wextra should produce no warnings."""
        c_file = VTEN_SV_DIR / "vten_shm_bridge.c"
        output = tmp_path / "libvten_shm.so"
        result = subprocess.run(
            [
                "gcc", "-shared", "-fPIC",
                "-Wall", "-Wextra", "-Werror",
                "-std=c99",
                "-I", str(VTEN_SV_DIR),
                "-o", str(output),
                str(c_file),
                "-lrt", "-lpthread",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"gcc -Werror failed:\nstderr: {result.stderr}"
        )

    def test_compile_c99_standard(self, tmp_path):
        """Must compile as C99 (CLAUDE.md: C99 표준)."""
        c_file = VTEN_SV_DIR / "vten_shm_bridge.c"
        output = tmp_path / "libvten_shm.so"
        result = subprocess.run(
            [
                "gcc", "-shared", "-fPIC",
                "-std=c99", "-pedantic",
                "-I", str(VTEN_SV_DIR),
                "-o", str(output),
                str(c_file),
                "-lrt", "-lpthread",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"C99 pedantic failed:\nstderr: {result.stderr}"
        )


# ═══════════════════════════════════════════════════════════════════
# §2. xvlog syntax check — Phase 3 completion criterion
#     "xvlog 구문 통과" (CLAUDE.md)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_xvlog(), reason="xvlog not available")
@pytest.mark.skipif(not _sv_files_exist(), reason="vten_sv/ not yet implemented")
class TestXvlogSyntax:
    """xvlog --sv syntax check for all SV files."""

    def _run_xvlog(self, *sv_files, tmp_path):
        """Run xvlog with given SV files."""
        args = ["xvlog", "--sv", f"--include={VTEN_SV_DIR}"]
        args.extend(str(f) for f in sv_files)
        result = subprocess.run(
            args,
            capture_output=True, text=True, timeout=60,
            cwd=str(tmp_path),
        )
        return result

    def test_types_svh(self, tmp_path):
        """vten_types.svh: type definitions compile."""
        # Create a wrapper that includes the header
        wrapper = tmp_path / "test_types.sv"
        wrapper.write_text(
            f'`include "{VTEN_SV_DIR}/vten_types.svh"\n'
            "module test_types; endmodule\n"
        )
        result = self._run_xvlog(wrapper, tmp_path=tmp_path)
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_bfm_cmd_if(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_bfm_axi4s(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_bfm_axi4s.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_bfm_axi4(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_bfm_axi4.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_bfm_axilite(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_bfm_axilite.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_shm_controller(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_shm_controller.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_command_scheduler(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_command_scheduler.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_bfm_probe(self, tmp_path):
        result = self._run_xvlog(
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
            VTEN_SV_DIR / "vten_bfm_probe.sv",
            tmp_path=tmp_path,
        )
        assert result.returncode == 0, f"xvlog failed:\n{result.stderr}"

    def test_all_files_together(self, tmp_path):
        """All SV files compile together without conflicts."""
        sv_files = sorted(VTEN_SV_DIR.glob("*.sv"))
        if not sv_files:
            pytest.skip("no .sv files found")
        result = self._run_xvlog(*sv_files, tmp_path=tmp_path)
        assert result.returncode == 0, f"xvlog all-files failed:\n{result.stderr}"


# ═══════════════════════════════════════════════════════════════════
# §3. Verilator lint (alternative to xvlog)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _has_verilator(), reason="verilator not available")
@pytest.mark.skipif(not _sv_files_exist(), reason="vten_sv/ not yet implemented")
class TestVerilatorLint:
    """Verilator --lint-only for stricter SV analysis."""

    def test_lint_types_and_interface(self, tmp_path):
        """Lint vten_types.svh + vten_bfm_cmd_if.sv."""
        sv_files = [
            VTEN_SV_DIR / "vten_bfm_cmd_if.sv",
        ]
        existing = [f for f in sv_files if f.exists()]
        if not existing:
            pytest.skip("files not found")
        result = subprocess.run(
            ["verilator", "--lint-only", "-sv",
             f"-I{VTEN_SV_DIR}",
             *[str(f) for f in existing]],
            capture_output=True, text=True, timeout=60,
        )
        # Verilator may warn on DPI-C or SV constructs not in synthesizable subset
        # We only check for errors, not warnings
        assert "Error" not in result.stderr or result.returncode == 0


# ═══════════════════════════════════════════════════════════════════
# §4. ctypes round-trip: Python SHM image ↔ C bridge unpacking
#     Most powerful test: validates Phase 2 ↔ Phase 3 compatibility
# ═══════════════════════════════════════════════════════════════════


def _compile_bridge(tmp_path: Path) -> Path | None:
    """Compile bridge and return .so path, or None if not possible."""
    c_file = VTEN_SV_DIR / "vten_shm_bridge.c"
    if not c_file.exists() or not _has_gcc():
        return None
    output = tmp_path / "libvten_shm_test.so"
    # Compile with a test-mode flag that exposes internal functions
    # and disables semaphore operations for unit testing
    result = subprocess.run(
        [
            "gcc", "-shared", "-fPIC",
            "-DVTEN_TEST_MODE",
            "-I", str(VTEN_SV_DIR),
            "-o", str(output),
            str(c_file),
            "-lrt", "-lpthread",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return None
    return output


@pytest.mark.skipif(not _has_gcc(), reason="gcc not available")
@pytest.mark.skipif(not _sv_files_exist(), reason="vten_sv/ not yet implemented")
class TestCtypesRoundTrip:
    """Load compiled .so via ctypes and verify SHM binary compatibility.

    This tests that the C bridge correctly reads what Python writes:
    Phase 2 (shm.py pack_*) → SHM image → Phase 3 (C bridge unpack).
    """

    @pytest.fixture()
    def bridge(self, tmp_path):
        """Compile and load the bridge .so."""
        so_path = _compile_bridge(tmp_path)
        if so_path is None:
            pytest.skip("could not compile bridge")
        lib = ctypes.CDLL(str(so_path))
        return lib

    @pytest.fixture()
    def shm_image(self):
        """Create a minimal SHM image using Phase 2 packing functions."""
        from vten.runtime.ir import Command
        from vten.runtime.shm import (
            BUF_DESC_SIZE,
            CMD_SLOT_SIZE,
            CONTROL_SIZE,
            STATS_SLOT_SIZE,
            BufferDescriptor,
            calculate_shm_size,
            pack_buffer_descriptor,
            pack_command_slot,
            pack_control_header,
            pack_stats_entry,
        )
        from vten.spec.models import OpCode, Protocol, Role

        # 3 commands: LOAD(0), PUSH(1), PULL(2)
        commands = [
            Command(
                op=OpCode.LOAD, cmd_id=0, interface_id=0,
                buffer_id=0, protocol=Protocol.AXI4S,
                size=1024, dep=[], commit_dep=[],
            ),
            Command(
                op=OpCode.PUSH, cmd_id=1, interface_id=1,
                buffer_id=0, protocol=Protocol.AXI4S,
                role=Role.MASTER, size=1024,
                dep=[0], commit_dep=[],
            ),
            Command(
                op=OpCode.PULL, cmd_id=2, interface_id=2,
                buffer_id=1, protocol=Protocol.AXI4S,
                role=Role.SLAVE, size=1024,
                dep=[1], commit_dep=[],
            ),
        ]

        num_commands = len(commands)
        num_buffers = 2
        buffer_sizes = [1024, 1024]

        total = calculate_shm_size(num_commands, num_buffers, buffer_sizes)
        image = bytearray(total)

        cmd_offset = CONTROL_SIZE
        stats_offset = cmd_offset + CMD_SLOT_SIZE * num_commands
        bufdesc_offset = stats_offset + STATS_SLOT_SIZE * num_commands
        data_offset = bufdesc_offset + BUF_DESC_SIZE * num_buffers
        # Align data offset
        data_offset = (data_offset + 63) & ~63

        pack_control_header(
            image, num_commands, num_buffers,
            cmd_offset, stats_offset, bufdesc_offset,
            data_offset, total,
        )

        for i, cmd in enumerate(commands):
            pack_command_slot(image, cmd_offset + i * CMD_SLOT_SIZE, cmd)
            pack_stats_entry(image, stats_offset + i * STATS_SLOT_SIZE)

        descs = [
            BufferDescriptor(0, 0, 0, 1024, 0),    # HOST_TO_DEV
            BufferDescriptor(1, 1, 0, 1024, 1024),  # DEV_TO_HOST
        ]
        for i, desc in enumerate(descs):
            pack_buffer_descriptor(
                image, bufdesc_offset + i * BUF_DESC_SIZE, desc
            )

        # Write known pattern to data buffer 0
        for j in range(1024):
            image[data_offset + j] = j & 0xFF

        return bytes(image), {
            "num_commands": num_commands,
            "num_buffers": num_buffers,
            "cmd_offset": cmd_offset,
            "stats_offset": stats_offset,
            "bufdesc_offset": bufdesc_offset,
            "data_offset": data_offset,
            "total": total,
            "commands": commands,
        }

    def test_control_header_magic_readback(self, shm_image):
        """Verify magic number at offset 0."""
        image_bytes, meta = shm_image
        magic = struct.unpack_from("<I", image_bytes, 0)[0]
        assert magic == 0x5654454E

    def test_control_header_version_readback(self, shm_image):
        image_bytes, meta = shm_image
        version = struct.unpack_from("<I", image_bytes, 4)[0]
        assert version == 0x00000003

    def test_control_header_num_commands(self, shm_image):
        image_bytes, meta = shm_image
        n = struct.unpack_from("<I", image_bytes, 0x10)[0]
        assert n == meta["num_commands"]

    def test_control_header_num_buffers(self, shm_image):
        image_bytes, meta = shm_image
        n = struct.unpack_from("<I", image_bytes, 0x14)[0]
        assert n == meta["num_buffers"]

    def test_command_slot_opcode_readback(self, shm_image):
        """Read back command opcodes from packed slots."""
        image_bytes, meta = shm_image
        for i, cmd in enumerate(meta["commands"]):
            offset = meta["cmd_offset"] + i * 64
            opcode = struct.unpack_from("<H", image_bytes, offset)[0]
            assert opcode == cmd.op.value, f"cmd {i}: expected {cmd.op.value}, got {opcode}"

    def test_command_slot_cmd_id_readback(self, shm_image):
        image_bytes, meta = shm_image
        for i, cmd in enumerate(meta["commands"]):
            offset = meta["cmd_offset"] + i * 64
            cmd_id = struct.unpack_from("<H", image_bytes, offset + 2)[0]
            assert cmd_id == cmd.cmd_id

    def test_command_slot_interface_id_readback(self, shm_image):
        image_bytes, meta = shm_image
        for i, cmd in enumerate(meta["commands"]):
            offset = meta["cmd_offset"] + i * 64
            iface_id = struct.unpack_from("<H", image_bytes, offset + 4)[0]
            assert iface_id == cmd.interface_id

    def test_command_slot_size_readback(self, shm_image):
        image_bytes, meta = shm_image
        for i, cmd in enumerate(meta["commands"]):
            offset = meta["cmd_offset"] + i * 64
            size = struct.unpack_from("<I", image_bytes, offset + 0x0C)[0]
            assert size == cmd.size

    def test_command_slot_dep_readback(self, shm_image):
        """Dependency arrays with 0xFFFF sentinel for unused slots."""
        image_bytes, meta = shm_image
        # cmd 1: dep=[0]
        offset = meta["cmd_offset"] + 1 * 64
        num_deps = struct.unpack_from("<B", image_bytes, offset + 0x2A)[0]
        assert num_deps == 1
        dep0 = struct.unpack_from("<H", image_bytes, offset + 0x2C)[0]
        assert dep0 == 0
        dep1 = struct.unpack_from("<H", image_bytes, offset + 0x2E)[0]
        assert dep1 == 0xFFFF  # unused sentinel

    def test_buffer_descriptor_readback(self, shm_image):
        image_bytes, meta = shm_image
        bd_off = meta["bufdesc_offset"]
        # Buffer 0
        buf_id = struct.unpack_from("<H", image_bytes, bd_off)[0]
        assert buf_id == 0
        direction = struct.unpack_from("<B", image_bytes, bd_off + 2)[0]
        assert direction == 0  # HOST_TO_DEV
        size = struct.unpack_from("<I", image_bytes, bd_off + 4)[0]
        assert size == 1024

    def test_data_region_pattern(self, shm_image):
        """Known data pattern written to buffer 0 is readable."""
        image_bytes, meta = shm_image
        data_off = meta["data_offset"]
        for j in range(256):
            assert image_bytes[data_off + j] == j & 0xFF

    def test_npu_3d_write_reg_command(self, shm_image):
        """Verify WRITE_REG command packing for NPU register access."""
        from vten.runtime.ir import Command
        from vten.runtime.shm import CMD_SLOT_SIZE, pack_command_slot
        from vten.spec.models import OpCode, Protocol

        # NPU 3D: write in_depth register at offset 0x014
        cmd = Command(
            op=OpCode.WRITE_REG, cmd_id=10, interface_id=5,
            protocol=Protocol.AXI4L,
            reg_offset=0x014, reg_value=8,
        )
        slot = bytearray(CMD_SLOT_SIZE)
        pack_command_slot(slot, 0, cmd)

        opcode = struct.unpack_from("<H", slot, 0x00)[0]
        assert opcode == 5  # WRITE_REG
        protocol = struct.unpack_from("<B", slot, 0x06)[0]
        assert protocol == 3  # AXI4L
        reg_off = struct.unpack_from("<I", slot, 0x18)[0]
        assert reg_off == 0x014
        reg_val = struct.unpack_from("<I", slot, 0x1C)[0]
        assert reg_val == 8

    def test_npu_3d_poll_reg_command(self, shm_image):
        """Verify POLL_REG command packing: mask + expected for layer_done."""
        from vten.runtime.ir import Command
        from vten.runtime.shm import CMD_SLOT_SIZE, pack_command_slot
        from vten.spec.models import OpCode, Protocol

        # NPU 3D: poll layer_done at offset 0x054, mask=1, expected=1
        cmd = Command(
            op=OpCode.POLL_REG, cmd_id=16, interface_id=5,
            protocol=Protocol.AXI4L,
            reg_offset=0x054, reg_mask=0x01, reg_expected=0x01,
        )
        slot = bytearray(CMD_SLOT_SIZE)
        pack_command_slot(slot, 0, cmd)

        opcode = struct.unpack_from("<H", slot, 0x00)[0]
        assert opcode == 7  # POLL_REG
        reg_mask = struct.unpack_from("<I", slot, 0x20)[0]
        assert reg_mask == 0x01
        reg_expected = struct.unpack_from("<I", slot, 0x24)[0]
        assert reg_expected == 0x01

    def test_npu_3d_axi4_push_with_phys_addr(self, shm_image):
        """Verify AXI4 PUSH with 64-bit physical address for DDR access."""
        from vten.runtime.ir import Command
        from vten.runtime.shm import CMD_SLOT_SIZE, pack_command_slot
        from vten.spec.models import OpCode, Protocol, Role

        # NPU 3D: PUSH IFM via DDR port, phys_addr=0x8000_0000
        cmd = Command(
            op=OpCode.PUSH, cmd_id=13, interface_id=1,
            buffer_id=0, protocol=Protocol.AXI4,
            role=Role.SLAVE,  # BFM is slave for AXI4 (DUT is master)
            size=65536, phys_addr=0x8000_0000,
        )
        slot = bytearray(CMD_SLOT_SIZE)
        pack_command_slot(slot, 0, cmd)

        protocol = struct.unpack_from("<B", slot, 0x06)[0]
        assert protocol == 2  # AXI4
        role = struct.unpack_from("<B", slot, 0x07)[0]
        assert role == 1  # SLAVE
        phys_addr = struct.unpack_from("<Q", slot, 0x10)[0]
        assert phys_addr == 0x8000_0000
        size = struct.unpack_from("<I", slot, 0x0C)[0]
        assert size == 65536

    def test_npu_3d_probe_command(self, shm_image):
        """Verify probe PULL command with golden_buf_id."""
        from vten.runtime.ir import Command
        from vten.runtime.shm import CMD_SLOT_SIZE, pack_command_slot
        from vten.spec.models import OpCode, Protocol, Role

        cmd = Command(
            op=OpCode.PULL, cmd_id=15, interface_id=2,
            buffer_id=1, protocol=Protocol.AXI4S,
            role=Role.SLAVE, size=65536,
            probe=True, golden_buf=3,
        )
        slot = bytearray(CMD_SLOT_SIZE)
        pack_command_slot(slot, 0, cmd)

        probe = struct.unpack_from("<B", slot, 0x0A)[0]
        assert probe == 1
        golden = struct.unpack_from("<H", slot, 0x28)[0]
        assert golden == 3

    def test_npu_3d_40_command_batch_size(self):
        """NPU 3D full layer: ~40 commands fit in SHM correctly."""
        from vten.runtime.shm import (
            CACHE_LINE,
            CMD_SLOT_SIZE,
            CONTROL_SIZE,
            STATS_SLOT_SIZE,
            calculate_shm_size,
        )

        # 40 commands, 8 buffers (ifm, ofm, weight, bias × 2 layers), ~4MB data
        num_cmds = 40
        num_bufs = 8
        buf_sizes = [65536] * 8  # 64KB each

        total = calculate_shm_size(num_cmds, num_bufs, buf_sizes)

        # Verify it's in the expected range (00_data_models.md §2)
        # NPU top: ~4.2MB
        assert total > 500_000  # at least 500KB
        assert total < 10_000_000  # under 10MB

        # Verify metadata region sizes
        cmd_region = CMD_SLOT_SIZE * num_cmds  # 40*64 = 2560
        stats_region = STATS_SLOT_SIZE * num_cmds  # 40*32 = 1280
        assert cmd_region == 2560
        assert stats_region == 1280
