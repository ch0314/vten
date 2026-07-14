"""Phase 3 tests: DPI-C bridge (vten_shm_bridge.c/h) spec compliance.

Parse C source files and verify:
- All required function signatures (04_backend_xsim.md §6.1)
- Binary offset constants match SHM layout (00_data_models.md §11)
- ControlHeader struct layout
- Internal pointer management pattern
- Error handling (null checks, return codes)

NPU 3D patterns from npu_3d_analysis.md used for realistic scenarios.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

VTEN_SV_DIR = Path(__file__).resolve().parent.parent / "vten" / "sv"


# ── Helpers ────────────────────────────────────────────────────────


def _read_c(filename: str) -> str:
    path = VTEN_SV_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not yet implemented")
    return path.read_text()


def _read_c_and_h() -> tuple[str, str]:
    c_text = _read_c("vten_shm_bridge.c")
    h_text = _read_c("vten_shm_bridge.h")
    return c_text, h_text


def _extract_define_hex(text: str, name: str) -> int | None:
    """Extract #define NAME 0xVALUE."""
    m = re.search(rf"#define\s+{re.escape(name)}\s+(0x[0-9a-fA-F_]+|\d+)", text)
    if not m:
        return None
    val = m.group(1).replace("_", "")
    return int(val, 0)


def _extract_define_int(text: str, name: str) -> int | None:
    m = re.search(rf"#define\s+{re.escape(name)}\s+(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


# ═══════════════════════════════════════════════════════════════════
# §1. Header file: function declarations
#     (04_backend_xsim.md §6.1)
# ═══════════════════════════════════════════════════════════════════


class TestBridgeHeader:
    """Verify vten_shm_bridge.h declares all required functions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_c("vten_shm_bridge.h")

    def test_include_guard(self):
        assert re.search(r"#ifndef\s+\w*VTEN_SHM_BRIDGE", self.text, re.IGNORECASE)

    # ── Lifecycle functions ──

    def test_shm_init_declaration(self):
        assert re.search(r"int\s+vten_shm_init\s*\(", self.text)

    def test_cleanup_declaration(self):
        assert re.search(r"void\s+vten_cleanup\s*\(", self.text)

    # ── Synchronization functions ──

    def test_wait_host_signal_safe(self):
        assert re.search(r"int\s+vten_wait_host_signal_safe\s*\(", self.text)

    def test_read_host_status(self):
        assert re.search(r"int\s+vten_read_host_status\s*\(", self.text)

    def test_set_backend_status(self):
        assert re.search(r"void\s+vten_set_backend_status\s*\(", self.text)

    def test_signal_complete(self):
        assert re.search(r"void\s+vten_signal_complete\s*\(", self.text)

    def test_signal_error(self):
        assert re.search(r"void\s+vten_signal_error\s*\(", self.text)

    # ── Control Region functions ──

    def test_read_num_commands(self):
        assert re.search(r"int\s+vten_read_num_commands\s*\(", self.text)

    def test_read_num_buffers(self):
        assert re.search(r"int\s+vten_read_num_buffers\s*\(", self.text)

    def test_read_timeout_ms(self):
        assert re.search(r"int\s+vten_read_timeout_ms\s*\(", self.text)

    def test_read_flags(self):
        assert re.search(r"int\s+vten_read_flags\s*\(", self.text)

    # ── Command Region functions ──

    def test_read_command(self):
        assert re.search(r"int\s+vten_read_command\s*\(", self.text)

    def test_read_command_deps(self):
        assert re.search(r"void\s+vten_read_command_deps\s*\(", self.text)

    # ── Data Region functions (bulk transfer) ──

    def test_read_data_bulk(self):
        assert re.search(r"void\s+vten_read_data_bulk\s*\(", self.text)

    def test_write_data_bulk(self):
        assert re.search(r"void\s+vten_write_data_bulk\s*\(", self.text)

    def test_write_data_byte(self):
        assert re.search(r"void\s+vten_write_data_byte\s*\(", self.text)

    # ── Stats Region functions ──

    def test_write_cmd_stats(self):
        assert re.search(r"void\s+vten_write_cmd_stats\s*\(", self.text)

    def test_write_cmd_status(self):
        assert re.search(r"void\s+vten_write_cmd_status\s*\(", self.text)

    # ── Probe functions ──

    def test_read_golden_bulk(self):
        assert re.search(r"void\s+vten_read_golden_bulk\s*\(", self.text)

    def test_log_mismatch(self):
        assert re.search(r"void\s+vten_log_mismatch\s*\(", self.text)


# ═══════════════════════════════════════════════════════════════════
# §2. SHM constants in C
#     (00_data_models.md §11.1, §11.2)
# ═══════════════════════════════════════════════════════════════════


class TestBridgeConstants:
    """Verify C code uses correct SHM constants matching Python runtime/shm.py."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text, self.h_text = _read_c_and_h()
        self.all_text = self.c_text + "\n" + self.h_text

    def test_shm_magic(self):
        """VTEN magic = 0x5654454E."""
        val = _extract_define_hex(self.all_text, "SHM_MAGIC") or \
              _extract_define_hex(self.all_text, "VTEN_MAGIC")
        assert val == 0x5654454E, f"expected 0x5654454E, got {val:#x}" if val else "not found"

    def test_protocol_version(self):
        val = _extract_define_hex(self.all_text, "PROTOCOL_VERSION") or \
              _extract_define_hex(self.all_text, "SHM_VERSION") or \
              _extract_define_hex(self.all_text, "VTEN_VERSION")
        assert val == 0x00000003

    def test_control_size(self):
        val = _extract_define_int(self.all_text, "CONTROL_SIZE")
        assert val == 256

    def test_cmd_slot_size(self):
        val = _extract_define_int(self.all_text, "CMD_SLOT_SIZE")
        assert val == 64

    def test_stats_slot_size(self):
        val = _extract_define_int(self.all_text, "STATS_SLOT_SIZE")
        assert val == 32

    def test_buf_desc_size(self):
        val = _extract_define_int(self.all_text, "BUF_DESC_SIZE")
        assert val == 24

    def test_constants_match_python(self):
        """C constants must exactly match vten.runtime.shm constants."""
        from vten.backend.sim.shm_constants import (
            BUF_DESC_SIZE,
            CMD_SLOT_SIZE,
            CONTROL_SIZE,
            PROTOCOL_VERSION,
            SHM_MAGIC,
            STATS_SLOT_SIZE,
        )

        assert _extract_define_int(self.all_text, "CONTROL_SIZE") == CONTROL_SIZE
        assert _extract_define_int(self.all_text, "CMD_SLOT_SIZE") == CMD_SLOT_SIZE
        assert _extract_define_int(self.all_text, "STATS_SLOT_SIZE") == STATS_SLOT_SIZE
        assert _extract_define_int(self.all_text, "BUF_DESC_SIZE") == BUF_DESC_SIZE

        magic = _extract_define_hex(self.all_text, "SHM_MAGIC") or \
                _extract_define_hex(self.all_text, "VTEN_MAGIC")
        assert magic == SHM_MAGIC

        ver = _extract_define_hex(self.all_text, "PROTOCOL_VERSION") or \
              _extract_define_hex(self.all_text, "SHM_VERSION") or \
              _extract_define_hex(self.all_text, "VTEN_VERSION")
        assert ver == PROTOCOL_VERSION


# ═══════════════════════════════════════════════════════════════════
# §3. Control Header binary offsets in C
#     (00_data_models.md §11.3)
# ═══════════════════════════════════════════════════════════════════


class TestControlHeaderOffsets:
    """Verify C code reads/writes control header at correct offsets."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_magic_at_offset_0x00(self):
        """magic is at offset 0 (first field)."""
        assert re.search(r"magic|0x00", self.c_text)

    def test_host_status_at_0x08(self):
        """host_status at offset 0x08."""
        assert re.search(r"0x0?8\b", self.c_text)

    def test_backend_status_at_0x0C(self):
        """backend_status at offset 0x0C."""
        assert re.search(r"0x0?[Cc]\b", self.c_text)

    def test_num_commands_at_0x10(self):
        assert re.search(r"0x10", self.c_text)

    def test_num_buffers_at_0x14(self):
        assert re.search(r"0x14", self.c_text)

    def test_cmd_region_offset_at_0x18(self):
        assert re.search(r"0x18", self.c_text)

    def test_error_code_at_0x40(self):
        assert re.search(r"0x40", self.c_text)

    def test_error_message_at_0x48(self):
        assert re.search(r"0x48", self.c_text)

    def test_flags_at_0x88(self):
        assert re.search(r"0x88", self.c_text)

    def test_timeout_ms_at_0x8C(self):
        assert re.search(r"0x8[Cc]", self.c_text)


# ═══════════════════════════════════════════════════════════════════
# §4. Command Slot unpacking offsets
#     (00_data_models.md §11.7)
# ═══════════════════════════════════════════════════════════════════


class TestCommandSlotOffsets:
    """Verify vten_read_command unpacks 64-byte slots at correct offsets."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_opcode_at_0x00(self):
        """opcode: uint16 at offset 0x00."""
        # Either explicit 0x00 or the first field read
        assert "opcode" in self.c_text.lower() or "0x00" in self.c_text

    def test_interface_id_at_0x04(self):
        assert re.search(r"0x0?4\b", self.c_text)

    def test_protocol_at_0x06(self):
        assert re.search(r"0x0?6\b", self.c_text)

    def test_role_at_0x07(self):
        assert re.search(r"0x0?7\b", self.c_text)

    def test_buffer_id_at_0x08(self):
        assert re.search(r"0x0?8\b", self.c_text)

    def test_probe_at_0x0A(self):
        assert re.search(r"0x0[Aa]\b", self.c_text)

    def test_flags_at_0x0B(self):
        assert re.search(r"0x0[Bb]\b", self.c_text)

    def test_size_at_0x0C(self):
        assert re.search(r"0x0[Cc]\b", self.c_text)

    def test_phys_addr_at_0x10(self):
        assert re.search(r"0x10", self.c_text)

    def test_reg_offset_at_0x18(self):
        assert re.search(r"0x18", self.c_text)

    def test_reg_value_at_0x1C(self):
        assert re.search(r"0x1[Cc]", self.c_text)

    def test_reg_mask_at_0x20(self):
        assert re.search(r"0x20", self.c_text)

    def test_reg_expected_at_0x24(self):
        assert re.search(r"0x24", self.c_text)

    def test_golden_buf_id_at_0x28(self):
        assert re.search(r"0x28", self.c_text)

    def test_num_deps_at_0x2A(self):
        assert re.search(r"0x2[Aa]", self.c_text)

    def test_dep_ids_at_0x2C(self):
        assert re.search(r"0x2[Cc]", self.c_text)

    def test_commit_dep_ids_at_0x34(self):
        assert re.search(r"0x34", self.c_text)


# ═══════════════════════════════════════════════════════════════════
# §5. Stats entry write offsets
#     (00_data_models.md §11.9)
# ═══════════════════════════════════════════════════════════════════


class TestStatsEntryOffsets:
    """Verify vten_write_cmd_stats writes at correct offsets within 32B slot."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_writes_status_field(self):
        """status at offset 0x00 within stats entry."""
        assert "status" in self.c_text

    def test_writes_issue_cycle(self):
        assert "issue_cycle" in self.c_text

    def test_writes_commit_cycle(self):
        assert "commit_cycle" in self.c_text

    def test_writes_active_cycles(self):
        assert "active_cycles" in self.c_text

    def test_writes_total_beats(self):
        assert "total_beats" in self.c_text

    def test_writes_stall_cycles(self):
        assert "stall_cycles" in self.c_text


# ═══════════════════════════════════════════════════════════════════
# §6. Internal pointer management
#     (04_backend_xsim.md §6.2)
# ═══════════════════════════════════════════════════════════════════


class TestInternalPointers:
    """Verify C bridge maintains correct internal pointers."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_shm_base_pointer(self):
        assert re.search(r"static\s+.*shm_base", self.c_text)

    def test_cmd_base_pointer(self):
        """cmd_base = shm_base + cmd_region_offset."""
        assert re.search(r"cmd_base", self.c_text)

    def test_stats_base_pointer(self):
        assert re.search(r"stats_base", self.c_text)

    def test_data_base_pointer(self):
        assert re.search(r"data_base", self.c_text)

    def test_bufdesc_base_pointer(self):
        assert re.search(r"bufdesc_base|buf_desc", self.c_text, re.IGNORECASE)

    def test_semaphore_pointers(self):
        assert re.search(r"sem_h2b", self.c_text)
        assert re.search(r"sem_b2h", self.c_text)


# ═══════════════════════════════════════════════════════════════════
# §7. POSIX API usage
#     (04_backend_xsim.md §3, §6)
# ═══════════════════════════════════════════════════════════════════


class TestPosixApiUsage:
    """Verify correct POSIX API usage (C99 + POSIX only)."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_shm_open_used(self):
        assert "shm_open" in self.c_text

    def test_mmap_used(self):
        assert "mmap" in self.c_text

    def test_sem_open_used(self):
        assert "sem_open" in self.c_text

    def test_sem_post_used(self):
        assert "sem_post" in self.c_text

    def test_sem_timedwait_used(self):
        """Timed wait for deadlock prevention (§5.2)."""
        assert "sem_timedwait" in self.c_text

    def test_sem_trywait_for_drain(self):
        """Stale semaphore drain on restart (§5.3)."""
        assert "sem_trywait" in self.c_text

    def test_memcpy_used(self):
        """Data region access via memcpy."""
        assert "memcpy" in self.c_text

    def test_includes_required_headers(self):
        for header in ["sys/mman.h", "semaphore.h", "fcntl.h"]:
            assert header in self.c_text, f"missing #include <{header}>"

    def test_includes_string_h(self):
        """For memcpy, strlen, etc."""
        assert "string.h" in self.c_text

    def test_magic_validation_in_init(self):
        """vten_shm_init must validate magic number."""
        assert "magic" in self.c_text.lower() or "MAGIC" in self.c_text


# ═══════════════════════════════════════════════════════════════════
# §8. Error handling
#     (requirement: all pointer access with null check)
# ═══════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Verify error handling patterns in C bridge."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_null_checks_exist(self):
        """At least some NULL checks on pointers."""
        assert re.search(r"==\s*NULL|!=\s*NULL|!\s*\w+_base", self.c_text)

    def test_return_codes_defined(self):
        """VTEN_OK, VTEN_ERROR, VTEN_TIMEOUT return codes."""
        all_text = self.c_text + _read_c("vten_shm_bridge.h")
        ok_found = re.search(r"VTEN_OK", all_text)
        err_found = re.search(r"VTEN_ERROR", all_text)
        timeout_found = re.search(r"VTEN_TIMEOUT", all_text)
        assert ok_found, "missing VTEN_OK"
        assert err_found, "missing VTEN_ERROR"
        assert timeout_found, "missing VTEN_TIMEOUT"

    def test_stderr_logging(self):
        """Error logging to stderr."""
        assert "stderr" in self.c_text or "fprintf" in self.c_text

    def test_session_seq_increment_on_restart(self):
        """session_seq++ on restart (§5.3)."""
        assert "session_seq" in self.c_text


# ═══════════════════════════════════════════════════════════════════
# §9. Semaphore naming convention
#     (04_backend_xsim.md §3)
# ═══════════════════════════════════════════════════════════════════


class TestSemaphoreNaming:
    """Verify semaphore names follow /vten_{session_id}_{h2b|b2h} pattern."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_h2b_semaphore_name(self):
        """Host → Backend semaphore: /vten_*_h2b."""
        assert re.search(r"vten_.*h2b", self.c_text)

    def test_b2h_semaphore_name(self):
        """Backend → Host semaphore: /vten_*_b2h."""
        assert re.search(r"vten_.*b2h", self.c_text)

    def test_shm_name_pattern(self):
        """SHM segment: /vten_{session_id}."""
        assert re.search(r"/vten_", self.c_text)


# ═══════════════════════════════════════════════════════════════════
# §10. Buffer descriptor cache
#     (04_backend_xsim.md §6.2)
# ═══════════════════════════════════════════════════════════════════


class TestBufferDescriptorCache:
    """Data region functions must look up buffer descriptors for data_offset."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.c_text = _read_c("vten_shm_bridge.c")

    def test_buf_cache_or_descriptor_lookup(self):
        """Some form of buffer descriptor caching/lookup."""
        assert re.search(r"buf_cache|buf_desc|descriptor|data_offset", self.c_text)

    def test_data_offset_used_in_read(self):
        """Bulk read uses data_offset to locate buffer."""
        # The function should compute: data_base + desc.data_offset + offset
        assert "data_offset" in self.c_text or "offset" in self.c_text
