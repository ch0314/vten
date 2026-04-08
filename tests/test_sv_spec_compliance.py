"""Phase 3 tests: SystemVerilog source spec compliance.

Parse vten_sv/ source files and verify they match the authoritative spec
(00_data_models.md, 04_backend_xsim.md, 05_bfm_library.md).

No SV simulator needed — pure source-level verification.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

VTEN_SV_DIR = Path(__file__).resolve().parent.parent / "vten_sv"


# ── Helpers ────────────────────────────────────────────────────────


def _read_sv(filename: str) -> str:
    """Read a SystemVerilog file from vten_sv/."""
    path = VTEN_SV_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not yet implemented")
    return path.read_text()


def _extract_localparam_int(text: str, name: str) -> int | None:
    """Extract `localparam int NAME = <value>;` from SV text."""
    pat = rf"localparam\s+(?:int\s+)?{re.escape(name)}\s*=\s*([^;]+);"
    m = re.search(pat, text)
    if not m:
        return None
    val = m.group(1).strip()
    # Handle hex: 32'h5654_454E or 32'hXXXX
    hex_match = re.search(r"'h([0-9a-fA-F_]+)", val)
    if hex_match:
        return int(hex_match.group(1).replace("_", ""), 16)
    # Handle decimal: 3'd1 or plain int
    dec_match = re.search(r"'d(\d+)", val)
    if dec_match:
        return int(dec_match.group(1))
    # Plain integer
    try:
        return int(val)
    except ValueError:
        return None


def _extract_enum_values(text: str, enum_name: str) -> dict[str, int]:
    """Extract enum members: NAME = N'd<value> or <hex>."""
    # Find enum block
    pat = rf"typedef\s+enum\s+[^{{]*\{{([^}}]+)\}}\s*{re.escape(enum_name)}\s*;"
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return {}
    body = m.group(1)
    result = {}
    for member_match in re.finditer(
        r"(\w+)\s*=\s*\d+'d(\d+)", body
    ):
        result[member_match.group(1)] = int(member_match.group(2))
    return result


# ═══════════════════════════════════════════════════════════════════
# §1. vten_types.svh — shared type definitions
#     (00_data_models.md §1, 04_backend_xsim.md §9.3)
# ═══════════════════════════════════════════════════════════════════


class TestVtenTypesSvh:
    """Verify vten_types.svh matches 00_data_models.md exactly."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_types.svh")

    # ── Include guard ──

    def test_include_guard(self):
        assert "`ifndef VTEN_TYPES_SVH" in self.text
        assert "`define VTEN_TYPES_SVH" in self.text
        assert "`endif" in self.text

    # ── OpCode enum (§1.4) ──

    def test_opcode_enum_exists(self):
        assert "opcode_t" in self.text

    def test_opcode_values(self):
        """OpCode values must match 00_data_models.md §1.4 exactly."""
        opcodes = _extract_enum_values(self.text, "opcode_t")
        expected = {
            "OP_LOAD": 1,
            "OP_PUSH": 2,
            "OP_PULL": 3,
            "OP_STORE": 4,
            "OP_WRITE_REG": 5,
            "OP_READ_REG": 6,
            "OP_POLL_REG": 7,
            "OP_BARRIER": 8,
        }
        for name, val in expected.items():
            assert opcodes.get(name) == val, f"{name}: expected {val}, got {opcodes.get(name)}"

    def test_opcode_count(self):
        """Exactly 8 opcodes."""
        opcodes = _extract_enum_values(self.text, "opcode_t")
        assert len(opcodes) == 8

    # ── Protocol enum (§1.1) ──

    def test_protocol_enum_exists(self):
        assert "protocol_t" in self.text

    def test_protocol_values(self):
        """Protocol SHM encoding: AXI4S=1, AXI4=2, AXI4L=3."""
        protos = _extract_enum_values(self.text, "protocol_t")
        assert protos.get("PROTO_AXI4S") == 1
        assert protos.get("PROTO_AXI4") == 2
        assert protos.get("PROTO_AXI4L") == 3

    # ── Role constants (§1.2) ──

    def test_role_master_is_zero(self):
        """MASTER=0, SLAVE=1."""
        assert re.search(r"ROLE_MASTER\s*=\s*1'b0", self.text)

    def test_role_slave_is_one(self):
        assert re.search(r"ROLE_SLAVE\s*=\s*1'b1", self.text)

    # ── CommandStatus enum (§1.7) ──

    def test_cmd_status_enum_exists(self):
        assert "cmd_status_t" in self.text

    def test_cmd_status_values(self):
        """PENDING=0, ISSUED=1, ACTIVE=2, COMMITTED=3, ERROR=4."""
        statuses = _extract_enum_values(self.text, "cmd_status_t")
        assert statuses.get("CMD_PENDING") == 0
        assert statuses.get("CMD_ISSUED") == 1
        assert statuses.get("CMD_ACTIVE") == 2
        assert statuses.get("CMD_COMMITTED") == 3
        assert statuses.get("CMD_ERROR") == 4

    # ── Backend/Host status constants (§11.4, §11.5) ──

    def test_backend_status_constants(self):
        assert _extract_localparam_int(self.text, "BACKEND_IDLE") == 0
        assert _extract_localparam_int(self.text, "BACKEND_RUNNING") == 1
        assert _extract_localparam_int(self.text, "BACKEND_DONE") == 2
        assert _extract_localparam_int(self.text, "BACKEND_ERROR") == 3

    def test_host_status_constants(self):
        assert _extract_localparam_int(self.text, "HOST_IDLE") == 0
        assert _extract_localparam_int(self.text, "HOST_CMD_READY") == 1
        assert _extract_localparam_int(self.text, "HOST_ACK") == 2
        assert _extract_localparam_int(self.text, "HOST_SHUTDOWN") == 3

    # ── Backend error codes (§11.13) ──

    def test_error_code_ok(self):
        assert _extract_localparam_int(self.text, "ERR_OK") == 0

    def test_error_code_addr_unmatch(self):
        assert _extract_localparam_int(self.text, "ERR_ADDR_UNMATCH") == 1

    def test_error_code_poll_timeout(self):
        assert _extract_localparam_int(self.text, "ERR_POLL_TIMEOUT") == 2

    def test_error_code_bfm_queue(self):
        assert _extract_localparam_int(self.text, "ERR_BFM_QUEUE") == 3

    def test_error_code_scheduler(self):
        assert _extract_localparam_int(self.text, "ERR_SCHEDULER") == 4

    def test_error_code_shm_access(self):
        assert _extract_localparam_int(self.text, "ERR_SHM_ACCESS") == 5

    def test_error_code_unknown_opcode(self):
        assert _extract_localparam_int(self.text, "ERR_UNKNOWN_OPCODE") == 6

    def test_error_code_bfm_map(self):
        assert _extract_localparam_int(self.text, "ERR_BFM_MAP") == 7

    def test_error_code_probe_mismatch(self):
        assert _extract_localparam_int(self.text, "ERR_PROBE_MISMATCH") == 8

    def test_error_code_timeout(self):
        assert _extract_localparam_int(self.text, "ERR_TIMEOUT") == 9

    # ── SHM constants (§11.1, §11.2) ──

    def test_shm_magic(self):
        """0x5654454E = "VTEN"."""
        assert _extract_localparam_int(self.text, "SHM_MAGIC") == 0x5654454E

    def test_shm_version(self):
        assert _extract_localparam_int(self.text, "SHM_VERSION") == 0x00000003

    def test_control_size(self):
        assert _extract_localparam_int(self.text, "CONTROL_SIZE") == 256

    def test_cmd_slot_size(self):
        assert _extract_localparam_int(self.text, "CMD_SLOT_SIZE") == 64

    def test_stats_slot_size(self):
        assert _extract_localparam_int(self.text, "STATS_SLOT_SIZE") == 32

    def test_buf_desc_size(self):
        assert _extract_localparam_int(self.text, "BUF_DESC_SIZE") == 24

    # ── DEP_NONE sentinel ──

    def test_dep_none_sentinel(self):
        """Unused dependency slot = 0xFFFF."""
        assert re.search(r"DEP_NONE\s*=\s*16'hFFFF", self.text)

    # ── bfm_cmd_t struct (§9.1) ──

    def test_bfm_cmd_t_exists(self):
        assert "bfm_cmd_t" in self.text

    def test_bfm_cmd_t_fields(self):
        """bfm_cmd_t must have all required fields."""
        required_fields = [
            "opcode", "cmd_id", "interface_id", "protocol", "role",
            "buffer_id", "probe", "sync", "size", "phys_addr",
            "reg_offset", "reg_value", "reg_mask", "reg_expected",
            "golden_buf_id",
        ]
        # Extract struct body
        m = re.search(
            r"typedef\s+struct\s+packed\s*\{([^}]+)\}\s*bfm_cmd_t",
            self.text, re.DOTALL,
        )
        assert m is not None, "bfm_cmd_t struct packed not found"
        body = m.group(1)
        for field in required_fields:
            assert re.search(rf"\b{field}\b", body), f"missing field: {field}"

    def test_bfm_cmd_t_no_dependency_fields(self):
        """Dependencies are NOT in bfm_cmd_t (Scheduler-only)."""
        m = re.search(
            r"typedef\s+struct\s+packed\s*\{([^}]+)\}\s*bfm_cmd_t",
            self.text, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "dep_ids" not in body
        assert "commit_dep" not in body
        assert "num_deps" not in body


# ═══════════════════════════════════════════════════════════════════
# §2. vten_bfm_cmd_if.sv — Scheduler ↔ BFM interface
#     (04_backend_xsim.md §9.2)
# ═══════════════════════════════════════════════════════════════════


class TestBfmCmdInterface:
    """Verify vten_bfm_cmd_if.sv interface declaration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_bfm_cmd_if.sv")

    def test_interface_keyword(self):
        assert re.search(r"\binterface\s+vten_bfm_cmd_if\b", self.text)

    def test_cmd_valid_signal(self):
        assert "cmd_valid" in self.text

    def test_cmd_data_signal(self):
        assert "cmd_data" in self.text

    def test_done_valid_signal(self):
        assert "done_valid" in self.text

    def test_done_cmd_id_signal(self):
        assert "done_cmd_id" in self.text

    def test_done_error_signal(self):
        assert "done_error" in self.text

    def test_done_error_code_signal(self):
        assert "done_error_code" in self.text

    def test_idle_signal(self):
        """v0.4.1: idle signal for all_drained calculation."""
        assert "idle" in self.text

    def test_scheduler_modport(self):
        assert re.search(r"modport\s+scheduler\b", self.text)

    def test_bfm_modport(self):
        assert re.search(r"modport\s+bfm\b", self.text)

    def test_includes_types(self):
        """Must include vten_types.svh."""
        assert "vten_types.svh" in self.text


# ═══════════════════════════════════════════════════════════════════
# §3. vten_bfm_axi4s.sv — AXI4-Stream BFM
#     (05_bfm_library.md §1)
# ═══════════════════════════════════════════════════════════════════


class TestBfmAxi4s:
    """Verify AXI4-Stream BFM module declaration and key features."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_bfm_axi4s.sv")

    def test_module_name(self):
        assert re.search(r"\bmodule\s+vten_bfm_axi4s\b", self.text)

    def test_data_w_parameter(self):
        assert re.search(r"parameter\s+.*DATA_W", self.text)

    def test_mode_parameter(self):
        """MODE = "MASTER" or "SLAVE"."""
        assert re.search(r'parameter\s+.*MODE\s*=\s*"MASTER"', self.text)

    # ── AXI4-Stream master ports ──

    def test_m_tdata_port(self):
        assert "m_tdata" in self.text

    def test_m_tvalid_port(self):
        assert "m_tvalid" in self.text

    def test_m_tready_port(self):
        assert "m_tready" in self.text

    def test_m_tlast_port(self):
        assert "m_tlast" in self.text

    # ── AXI4-Stream slave ports ──

    def test_s_tdata_port(self):
        assert "s_tdata" in self.text

    def test_s_tvalid_port(self):
        assert "s_tvalid" in self.text

    def test_s_tready_port(self):
        assert "s_tready" in self.text

    def test_s_tlast_port(self):
        assert "s_tlast" in self.text

    # ── Scheduler interface ──

    def test_cmd_if_port(self):
        assert "cmd_if" in self.text

    def test_cycle_count_port(self):
        assert "cycle_count" in self.text

    # ── Key implementation details ──

    def test_bytes_per_beat_calculation(self):
        """BYTES_PER_BEAT = DATA_W / 8."""
        assert "BYTES_PER_BEAT" in self.text

    def test_idle_signal(self):
        """v0.4.1: idle = !cmd_active && queue empty."""
        assert "idle" in self.text

    def test_dpi_read_data_call(self):
        """MASTER mode reads from SHM via vten_read_data."""
        assert "vten_read_data" in self.text

    def test_dpi_write_data_call(self):
        """SLAVE mode writes to SHM via vten_write_data."""
        assert "vten_write_data" in self.text

    def test_dpi_write_cmd_stats(self):
        """Stats recording on command completion."""
        assert "vten_write_cmd_stats" in self.text

    def test_probe_mode_support(self):
        """Probe mode: golden comparison in SLAVE mode."""
        assert "vten_read_golden" in self.text

    def test_mismatch_logging(self):
        assert "vten_log_mismatch" in self.text

    def test_stats_tracking_fields(self):
        """Must track: issue_cycle, first_active, last_active,
        active_cycles, stall_cycles, total_beats."""
        for field in ["issue_cycle", "first_active", "last_active",
                       "active_cycles", "stall_cycles", "total_beats"]:
            assert field in self.text, f"missing stats field: {field}"


# ═══════════════════════════════════════════════════════════════════
# §4. vten_bfm_axi4.sv — AXI4 Memory-Mapped BFM
#     (05_bfm_library.md §2)
# ═══════════════════════════════════════════════════════════════════


class TestBfmAxi4:
    """Verify AXI4 MM BFM: DUT=master, BFM=slave."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_bfm_axi4.sv")

    def test_module_name(self):
        assert re.search(r"\bmodule\s+vten_bfm_axi4\b", self.text)

    def test_data_w_parameter(self):
        assert re.search(r"parameter\s+.*DATA_W", self.text)

    def test_addr_w_parameter(self):
        assert re.search(r"parameter\s+.*ADDR_W", self.text)

    # ── AXI4 Read channel (AR + R) ──

    def test_ar_channel_ports(self):
        for port in ["s_araddr", "s_arlen", "s_arsize", "s_arburst",
                      "s_arvalid", "s_arready"]:
            assert port in self.text, f"missing AR port: {port}"

    def test_r_channel_ports(self):
        for port in ["s_rdata", "s_rresp", "s_rlast", "s_rvalid", "s_rready"]:
            assert port in self.text, f"missing R port: {port}"

    # ── AXI4 Write channel (AW + W + B) ──

    def test_aw_channel_ports(self):
        for port in ["s_awaddr", "s_awlen", "s_awsize", "s_awburst",
                      "s_awvalid", "s_awready"]:
            assert port in self.text, f"missing AW port: {port}"

    def test_w_channel_ports(self):
        for port in ["s_wdata", "s_wstrb", "s_wlast", "s_wvalid", "s_wready"]:
            assert port in self.text, f"missing W port: {port}"

    def test_b_channel_ports(self):
        for port in ["s_bresp", "s_bvalid", "s_bready"]:
            assert port in self.text, f"missing B port: {port}"

    # ── Key implementation details ──

    def test_active_table(self):
        """Multiple active commands for address matching."""
        assert "active_table" in self.text

    def test_find_entry_function(self):
        """Address matching function."""
        assert "find_entry" in self.text

    def test_done_queue(self):
        """v0.4.1: done_queue prevents same-cycle completion loss."""
        assert "done_queue" in self.text

    def test_idle_signal(self):
        """v0.4.1: idle when all queues/pending empty."""
        assert "idle" in self.text

    def test_decerr_response(self):
        """DECERR (2'b11) on address mismatch."""
        assert re.search(r"2'b11", self.text)

    def test_wstrb_handling(self):
        """Write strobe byte-selective write."""
        assert "s_wstrb" in self.text

    def test_check_completion(self):
        """Completion tracking when transferred_bytes >= expected."""
        assert "check_completion" in self.text

    def test_probe_mode(self):
        """Probe golden comparison in write path."""
        assert "vten_read_golden" in self.text


# ═══════════════════════════════════════════════════════════════════
# §5. vten_bfm_axilite.sv — AXI4-Lite BFM (master)
#     (05_bfm_library.md §3)
# ═══════════════════════════════════════════════════════════════════


class TestBfmAxiLite:
    """Verify AXI4-Lite BFM: BFM=master, drives register transactions."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_bfm_axilite.sv")

    def test_module_name(self):
        assert re.search(r"\bmodule\s+vten_bfm_axilite\b", self.text)

    def test_addr_w_parameter(self):
        assert re.search(r"parameter\s+.*ADDR_W\s*=\s*32", self.text)

    def test_data_w_parameter(self):
        assert re.search(r"parameter\s+.*DATA_W\s*=\s*32", self.text)

    # ── AXI4-Lite Master Write ports ──

    def test_aw_ports(self):
        for port in ["m_awaddr", "m_awvalid", "m_awready"]:
            assert port in self.text, f"missing: {port}"

    def test_w_ports(self):
        for port in ["m_wdata", "m_wstrb", "m_wvalid", "m_wready"]:
            assert port in self.text, f"missing: {port}"

    def test_b_ports(self):
        for port in ["m_bresp", "m_bvalid", "m_bready"]:
            assert port in self.text, f"missing: {port}"

    # ── AXI4-Lite Master Read ports ──

    def test_ar_ports(self):
        for port in ["m_araddr", "m_arvalid", "m_arready"]:
            assert port in self.text, f"missing: {port}"

    def test_r_ports(self):
        for port in ["m_rdata", "m_rresp", "m_rvalid", "m_rready"]:
            assert port in self.text, f"missing: {port}"

    # ── Operations ──

    def test_write_reg_operation(self):
        assert "OP_WRITE_REG" in self.text

    def test_read_reg_operation(self):
        assert "OP_READ_REG" in self.text

    def test_poll_reg_operation(self):
        assert "OP_POLL_REG" in self.text

    def test_poll_mask_and_expected(self):
        """POLL_REG: (rdata & mask) == expected."""
        assert "reg_mask" in self.text
        assert "reg_expected" in self.text

    def test_poll_timeout_error_code(self):
        """Poll timeout → BackendErrorCode.POLL_TIMEOUT = 2."""
        # Should reference error code 2 somewhere
        assert re.search(r"16'd2|ERR_POLL_TIMEOUT", self.text)

    def test_idle_signal(self):
        assert "idle" in self.text


# ═══════════════════════════════════════════════════════════════════
# §6. vten_shm_controller.sv — SHM state machine
#     (04_backend_xsim.md §4)
# ═══════════════════════════════════════════════════════════════════


class TestShmController:
    """Verify SHM Controller state machine and DPI-C integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_shm_controller.sv")

    def test_module_name(self):
        assert re.search(r"\bmodule\s+vten_shm_controller\b", self.text)

    def test_session_id_parameter(self):
        assert "SESSION_ID" in self.text

    def test_max_cmds_parameter(self):
        assert "MAX_CMDS" in self.text

    # ── State enum ──

    def test_all_states_defined(self):
        """All 9 states from spec §4."""
        for state in ["S_INIT", "S_WAIT_HOST", "S_LOAD_BATCH",
                       "S_FEED", "S_EXECUTE", "S_DRAIN",
                       "S_COMPLETE", "S_ERROR", "S_SHUTDOWN"]:
            assert state in self.text, f"missing state: {state}"

    # ── Controller ↔ Scheduler interface ──

    def test_feed_valid_output(self):
        assert "feed_valid" in self.text

    def test_feed_data_output(self):
        assert "feed_data" in self.text

    def test_feed_ready_input(self):
        assert "feed_ready" in self.text

    def test_feed_done_output(self):
        assert "feed_done" in self.text

    def test_sched_all_committed_input(self):
        assert "sched_all_committed" in self.text

    def test_sched_all_drained_input(self):
        assert "sched_all_drained" in self.text

    def test_sched_error_input(self):
        assert "sched_error" in self.text

    # ── DPI-C function calls ──

    def test_dpi_shm_init(self):
        assert "vten_shm_init" in self.text

    def test_dpi_cleanup(self):
        assert "vten_cleanup" in self.text

    def test_dpi_wait_host_signal(self):
        assert "vten_wait_host_signal_safe" in self.text

    def test_dpi_read_host_status(self):
        assert "vten_read_host_status" in self.text

    def test_dpi_set_backend_status(self):
        assert "vten_set_backend_status" in self.text

    def test_dpi_signal_complete(self):
        assert "vten_signal_complete" in self.text

    def test_dpi_signal_error(self):
        assert "vten_signal_error" in self.text

    def test_dpi_read_num_commands(self):
        assert "vten_read_num_commands" in self.text

    def test_dpi_read_command(self):
        assert "vten_read_command" in self.text

    # ── Key design decisions ──

    def test_single_always_ff(self):
        """All DPI-C calls in always_ff (not always_comb). v0.4.1 fix."""
        assert "always_ff" in self.text

    def test_cmd_cache(self):
        """Local command cache: SHM → cache in S_LOAD_BATCH."""
        assert "cmd_cache" in self.text

    def test_finish_in_shutdown(self):
        assert "$finish" in self.text


# ═══════════════════════════════════════════════════════════════════
# §7. vten_command_scheduler.sv — Command Scheduler
#     (04_backend_xsim.md §10)
# ═══════════════════════════════════════════════════════════════════


class TestCommandScheduler:
    """Verify Command Scheduler: dependency resolution and BFM dispatch."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_command_scheduler.sv")

    def test_module_name(self):
        assert re.search(r"\bmodule\s+vten_command_scheduler\b", self.text)

    def test_max_cmds_parameter(self):
        assert "MAX_CMDS" in self.text

    def test_max_bfms_parameter(self):
        assert "MAX_BFMS" in self.text

    def test_max_ifaces_parameter(self):
        assert "MAX_IFACES" in self.text

    # ── Controller ↔ Scheduler ports ──

    def test_feed_ports(self):
        for port in ["feed_valid", "feed_data", "feed_ready", "feed_done"]:
            assert port in self.text, f"missing: {port}"

    def test_status_outputs(self):
        assert "all_committed" in self.text
        assert "all_drained" in self.text

    def test_error_outputs(self):
        assert "error_flag" in self.text
        assert "error_cmd_id" in self.text
        assert "error_code" in self.text

    def test_iface_to_bfm_mapping(self):
        """interface_id → BFM index mapping input."""
        assert "iface_to_bfm" in self.text

    # ── Dependency tracking ──

    def test_issued_bitmap(self):
        assert "issued" in self.text

    def test_bfm_done_bitmap(self):
        assert "bfm_done" in self.text

    def test_committed_bitmap(self):
        assert "committed" in self.text

    def test_dependency_storage(self):
        """Separate dep/commit_dep storage (not in bfm_cmd_t)."""
        assert "cmd_dep" in self.text
        assert "cmd_commit_dep" in self.text

    def test_preprocess_batch(self):
        """Batch preprocessing: load deps, compute sync chain, barrier fence."""
        assert "preprocess_batch" in self.text

    def test_dpi_read_command_deps(self):
        """DPI-C call to extract dependencies."""
        assert "vten_read_command_deps" in self.text


# ═══════════════════════════════════════════════════════════════════
# §8. vten_dpi_imports.svh — DPI-C function declarations
#     (04_backend_xsim.md §6.1)
# ═══════════════════════════════════════════════════════════════════


class TestDpiImportsSvh:
    """Verify all required DPI-C functions are declared."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_dpi_imports.svh")

    def test_all_required_functions_declared(self):
        """All DPI-C functions from spec §6.1 must be declared."""
        required = [
            "vten_shm_init",
            "vten_cleanup",
            "vten_wait_host_signal_safe",
            "vten_read_host_status",
            "vten_set_backend_status",
            "vten_signal_complete",
            "vten_signal_error",
            "vten_read_num_commands",
            "vten_read_num_buffers",
            "vten_read_timeout_ms",
            "vten_read_flags",
            "vten_read_command",
            "vten_read_command_deps",
            "vten_read_data",
            "vten_write_data",
            "vten_write_cmd_stats",
            "vten_write_cmd_status",
            "vten_read_golden",
            "vten_log_mismatch",
        ]
        for func in required:
            assert func in self.text, f"missing DPI-C function: {func}"

    def test_import_dpi_c_syntax(self):
        """All functions use proper import "DPI-C" syntax."""
        count = len(re.findall(r'import\s+"DPI-C"', self.text))
        assert count >= 19, f"expected ≥19 DPI-C imports, found {count}"


# ═══════════════════════════════════════════════════════════════════
# §9. vten_bfm_probe.sv — Probe BFM
#     (05_bfm_library.md §4.2)
# ═══════════════════════════════════════════════════════════════════


class TestBfmProbe:
    """Verify Probe BFM for signal-level golden comparison."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.text = _read_sv("vten_bfm_probe.sv")

    def test_module_exists(self):
        assert re.search(r"\bmodule\s+vten_bfm.*probe\b", self.text, re.IGNORECASE)

    def test_data_w_parameter(self):
        assert re.search(r"parameter\s+.*DATA_W", self.text)

    def test_golden_comparison(self):
        assert "vten_read_golden" in self.text

    def test_mismatch_logging(self):
        assert "vten_log_mismatch" in self.text


# ═══════════════════════════════════════════════════════════════════
# §10. Cross-file consistency checks
# ═══════════════════════════════════════════════════════════════════


class TestCrossFileConsistency:
    """Verify SV files use consistent types and constants."""

    def test_all_bfms_include_types(self):
        """All BFM modules must include vten_types.svh."""
        for filename in ["vten_bfm_axi4s.sv", "vten_bfm_axi4.sv",
                          "vten_bfm_axilite.sv"]:
            text = _read_sv(filename)
            assert "vten_types.svh" in text, f"{filename} missing types include"

    def test_controller_includes_types(self):
        text = _read_sv("vten_shm_controller.sv")
        assert "vten_types.svh" in text

    def test_scheduler_includes_types(self):
        text = _read_sv("vten_command_scheduler.sv")
        assert "vten_types.svh" in text

    def test_all_sv_files_exist(self):
        """All 9 required SV files from CLAUDE.md project structure."""
        required = [
            "vten_types.svh",
            "vten_dpi_imports.svh",
            "vten_bfm_cmd_if.sv",
            "vten_shm_controller.sv",
            "vten_command_scheduler.sv",
            "vten_bfm_axi4s.sv",
            "vten_bfm_axi4.sv",
            "vten_bfm_axilite.sv",
            "vten_bfm_probe.sv",
        ]
        for f in required:
            path = VTEN_SV_DIR / f
            assert path.exists(), f"missing: vten_sv/{f}"

    def test_c_files_exist(self):
        """DPI-C bridge C files."""
        assert (VTEN_SV_DIR / "vten_shm_bridge.c").exists()
        assert (VTEN_SV_DIR / "vten_shm_bridge.h").exists()
