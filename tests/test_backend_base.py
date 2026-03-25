"""Phase 4 tests: Backend base — error codes, ABC, result types.

Spec references:
- 00_data_models.md §10.13 (BackendErrorCode)
- 00_data_models.md §12 (Error Taxonomy)
- 00_data_models.md §13 (BackendResult, BatchResult)
- 06_codegen_and_cli.md §5 (Error Propagation Path)
"""

from __future__ import annotations

import abc
from dataclasses import fields

import pytest

from vten.errors import BackendError, BFMError, PollTimeoutError
from vten.errors import TimeoutError as VTenTimeoutError


# ═══════════════════════════════════════════════════════════════════
# §1  BackendErrorCode — 00_data_models.md §10.13
# ═══════════════════════════════════════════════════════════════════


class TestBackendErrorCode:
    """BackendErrorCode enum: 10 error codes (OK=0 … TIMEOUT=9)."""

    def test_ok_is_zero(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.OK == 0

    def test_addr_unmatch_is_1(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.ADDR_UNMATCH == 1

    def test_poll_timeout_is_2(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.POLL_TIMEOUT == 2

    def test_bfm_queue_error_is_3(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.BFM_QUEUE_ERROR == 3

    def test_scheduler_error_is_4(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.SCHEDULER_ERROR == 4

    def test_shm_access_error_is_5(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.SHM_ACCESS_ERROR == 5

    def test_unknown_opcode_is_6(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.UNKNOWN_OPCODE == 6

    def test_bfm_map_error_is_7(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.BFM_MAP_ERROR == 7

    def test_probe_mismatch_is_8(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.PROBE_MISMATCH == 8

    def test_timeout_is_9(self):
        from vten.backend.base import BackendErrorCode

        assert BackendErrorCode.TIMEOUT == 9

    def test_completeness_ten_codes(self):
        """All 10 codes from spec §10.13 are present."""
        from vten.backend.base import BackendErrorCode

        assert len(BackendErrorCode) == 10

    def test_all_values_are_unique(self):
        from vten.backend.base import BackendErrorCode

        values = [e.value for e in BackendErrorCode]
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════
# §2  Backend error code → exception mapping
# ═══════════════════════════════════════════════════════════════════


class TestBackendErrorMap:
    """Error code → exception mapping (06_codegen_and_cli.md §5)."""

    def test_addr_unmatch_raises_bfm_error(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BFMError):
            raise_backend_error(code=1, cmd_id=5, message="DECERR")

    def test_poll_timeout_raises_poll_timeout_error(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(PollTimeoutError):
            raise_backend_error(code=2, cmd_id=8, message="timeout")

    def test_bfm_queue_error_raises_bfm_error(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BFMError):
            raise_backend_error(code=3, cmd_id=0, message="queue")

    def test_timeout_raises_timeout_error(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(VTenTimeoutError):
            raise_backend_error(code=9, cmd_id=0, message="global timeout")

    def test_scheduler_error_raises_backend_error(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError):
            raise_backend_error(code=4, cmd_id=3, message="deadlock")

    def test_unknown_code_raises_backend_error(self):
        """Unmapped or generic codes fall back to base BackendError."""
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError):
            raise_backend_error(code=5, cmd_id=0, message="shm fail")

    def test_error_message_includes_cmd_id(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError, match="cmd_id=5"):
            raise_backend_error(code=1, cmd_id=5, message="DECERR")

    def test_error_message_includes_original_message(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError, match="DECERR"):
            raise_backend_error(code=1, cmd_id=5, message="DECERR")

    def test_error_has_context_fields(self):
        """Raised exception carries error_code and cmd_id in context."""
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError) as exc_info:
            raise_backend_error(code=2, cmd_id=7, message="poll fail")
        assert exc_info.value.context["error_code"] == 2
        assert exc_info.value.context["cmd_id"] == 7

    def test_error_message_format(self):
        """Error message follows spec format:
        '[<source>] <description> (<context>)'
        """
        from vten.backend.base import raise_backend_error

        with pytest.raises(BackendError) as exc_info:
            raise_backend_error(
                code=1, cmd_id=5,
                message="[BFM:data_port] DECERR at addr=0x00100000",
            )
        msg = str(exc_info.value)
        assert "BFM:data_port" in msg
        assert "DECERR" in msg


# ═══════════════════════════════════════════════════════════════════
# §3  Backend ABC — 00_data_models.md §13, 04_backend_xsim.md
# ═══════════════════════════════════════════════════════════════════


class TestBackendABC:
    """Backend abstract base class contract."""

    def test_cannot_instantiate_abstract(self):
        from vten.backend.base import Backend

        with pytest.raises(TypeError):
            Backend()  # type: ignore[abstract]

    def test_is_abstract_class(self):
        from vten.backend.base import Backend

        assert abc.ABC in Backend.__mro__

    def test_execute_is_abstract(self):
        from vten.backend.base import Backend

        assert "execute" in Backend.__abstractmethods__

    def test_cleanup_is_abstract(self):
        from vten.backend.base import Backend

        assert "cleanup" in Backend.__abstractmethods__

    def test_submit_not_abstract(self):
        """submit() is optional (non-abstract), raises NotImplementedError by default."""
        from vten.backend.base import Backend

        assert "submit" not in Backend.__abstractmethods__

    def test_wait_not_abstract(self):
        """wait() is optional (non-abstract), raises NotImplementedError by default."""
        from vten.backend.base import Backend

        assert "wait" not in Backend.__abstractmethods__

    def test_shutdown_not_abstract(self):
        """shutdown() is optional (non-abstract), default is pass."""
        from vten.backend.base import Backend

        assert "shutdown" not in Backend.__abstractmethods__

    def test_concrete_subclass_instantiates(self):
        """A complete concrete subclass can be instantiated."""
        from vten.backend.base import Backend

        class StubBackend(Backend):
            def execute(self, compiled):
                pass

            def cleanup(self):
                pass

        b = StubBackend()
        assert isinstance(b, Backend)

    def test_context_manager(self):
        """Backend supports with-statement via __enter__/__exit__."""
        from vten.backend.base import Backend

        class StubBackend(Backend):
            def __init__(self):
                self.cleaned = False

            def execute(self, compiled):
                pass

            def cleanup(self):
                self.cleaned = True

        with StubBackend() as b:
            assert isinstance(b, Backend)
        assert b.cleaned

    def test_submit_raises_not_implemented(self):
        from vten.backend.base import Backend

        class StubBackend(Backend):
            def execute(self, compiled):
                pass

            def cleanup(self):
                pass

        b = StubBackend()
        with pytest.raises(NotImplementedError):
            b.submit(None)

    def test_wait_raises_not_implemented(self):
        from vten.backend.base import Backend

        class StubBackend(Backend):
            def execute(self, compiled):
                pass

            def cleanup(self):
                pass

        b = StubBackend()
        with pytest.raises(NotImplementedError):
            b.wait()


# ═══════════════════════════════════════════════════════════════════
# §4  BackendResult — 00_data_models.md §13
# ═══════════════════════════════════════════════════════════════════


class TestBackendResult:
    """BackendResult: returned when backend_status == DONE (always status=2)."""

    def test_construction_done(self):
        from vten.backend.base import BackendResult

        r = BackendResult(status=2)
        assert r.status == 2
        assert r.error_code == 0
        assert r.error_cmd_id == 0
        assert r.error_message == ""

    def test_stats_default_empty_list(self):
        from vten.backend.base import BackendResult

        r = BackendResult(status=2)
        assert r.stats == []

    def test_stats_populated(self):
        from vten.backend.base import BackendResult, CmdStats

        s = CmdStats(
            cmd_id=0, status=3, issue_cycle=100, commit_cycle=200,
            first_active_cycle=110, last_active_cycle=190,
            active_cycles=80, total_beats=10, stall_cycles=0,
        )
        r = BackendResult(status=2, stats=[s])
        assert len(r.stats) == 1
        assert r.stats[0].cmd_id == 0

    def test_read_buffer_method_exists(self):
        from vten.backend.base import BackendResult

        r = BackendResult(status=2)
        assert hasattr(r, "read_buffer")

    def test_is_dataclass(self):
        from vten.backend.base import BackendResult

        assert hasattr(BackendResult, "__dataclass_fields__")

    def test_error_fields_default_zero(self):
        from vten.backend.base import BackendResult

        r = BackendResult(status=2)
        assert r.error_code == 0
        assert r.error_cmd_id == 0
        assert r.error_message == ""


# ═══════════════════════════════════════════════════════════════════
# §5  BatchResult — 00_data_models.md §13
# ═══════════════════════════════════════════════════════════════════


class TestBatchResult:
    """BatchResult: ctx.run() return value."""

    def test_construction_done(self):
        from vten.backend.base import BatchResult

        r = BatchResult(status="DONE", total_cycles=5000, per_command_stats=[])
        assert r.status == "DONE"
        assert r.total_cycles == 5000
        assert r.error is None

    def test_construction_error(self):
        from vten.backend.base import BatchResult

        err = BackendError("backend failed")
        r = BatchResult(
            status="ERROR", total_cycles=0,
            per_command_stats=[], error=err,
        )
        assert r.status == "ERROR"
        assert r.error is not None

    def test_error_is_none_on_success(self):
        from vten.backend.base import BatchResult

        r = BatchResult(status="DONE", total_cycles=100, per_command_stats=[])
        assert r.error is None

    def test_status_values(self):
        """Only DONE or ERROR are valid status values."""
        from vten.backend.base import BatchResult

        done = BatchResult(status="DONE", total_cycles=0, per_command_stats=[])
        error = BatchResult(
            status="ERROR", total_cycles=0, per_command_stats=[],
            error=BackendError("fail"),
        )
        assert done.status == "DONE"
        assert error.status == "ERROR"

    def test_per_command_stats_list(self):
        from vten.backend.base import BatchResult, CmdStats

        s = CmdStats(
            cmd_id=0, status=3, issue_cycle=10, commit_cycle=50,
            first_active_cycle=15, last_active_cycle=45,
            active_cycles=30, total_beats=4, stall_cycles=5,
        )
        r = BatchResult(status="DONE", total_cycles=100, per_command_stats=[s])
        assert len(r.per_command_stats) == 1

    def test_is_dataclass(self):
        from vten.backend.base import BatchResult

        assert hasattr(BatchResult, "__dataclass_fields__")


# ═══════════════════════════════════════════════════════════════════
# §6  CmdStats — Stats Region per-command metrics
# ═══════════════════════════════════════════════════════════════════


class TestCmdStats:
    """CmdStats: per-command execution statistics from Stats Region."""

    def test_construction(self):
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=100, commit_cycle=500,
            first_active_cycle=150, last_active_cycle=450,
            active_cycles=200, total_beats=32, stall_cycles=50,
        )
        assert s.cmd_id == 0
        assert s.status == 3
        assert s.issue_cycle == 100
        assert s.commit_cycle == 500

    def test_is_dataclass(self):
        from vten.backend.base import CmdStats

        assert hasattr(CmdStats, "__dataclass_fields__")

    def test_latency_cycles(self):
        """Latency = commit_cycle - issue_cycle."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=100, commit_cycle=500,
            first_active_cycle=150, last_active_cycle=450,
            active_cycles=200, total_beats=32, stall_cycles=50,
        )
        assert s.latency_cycles == 400

    def test_active_window(self):
        """Active window = last_active_cycle - first_active_cycle + 1."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=100, commit_cycle=500,
            first_active_cycle=150, last_active_cycle=450,
            active_cycles=200, total_beats=32, stall_cycles=50,
        )
        assert s.active_window == 301

    def test_utilization(self):
        """Utilization = active_cycles / active_window."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=100, commit_cycle=500,
            first_active_cycle=100, last_active_cycle=199,
            active_cycles=80, total_beats=10, stall_cycles=20,
        )
        # active_window = 100, active_cycles = 80 → 0.8
        assert abs(s.utilization - 0.8) < 1e-9

    def test_bus_efficiency(self):
        """Bus efficiency = active_cycles / latency_cycles."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=0, commit_cycle=100,
            first_active_cycle=10, last_active_cycle=90,
            active_cycles=50, total_beats=10, stall_cycles=40,
        )
        # latency = 100, active = 50 → 0.5
        assert abs(s.bus_efficiency - 0.5) < 1e-9

    def test_zero_active_window_returns_zero(self):
        """Edge case: first == last → window=1, not division by zero."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=0, commit_cycle=1,
            first_active_cycle=0, last_active_cycle=0,
            active_cycles=1, total_beats=1, stall_cycles=0,
        )
        # active_window = 1
        assert s.active_window == 1
        assert abs(s.utilization - 1.0) < 1e-9

    def test_zero_latency_returns_zero(self):
        """Edge case: issue == commit (instant). Metrics should not crash."""
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=0, status=3,
            issue_cycle=100, commit_cycle=100,
            first_active_cycle=100, last_active_cycle=100,
            active_cycles=0, total_beats=0, stall_cycles=0,
        )
        # latency = 0 → bus_efficiency should be 0 (no division by zero)
        assert s.latency_cycles == 0
        assert s.bus_efficiency == 0.0

    def test_all_fields_present(self):
        """CmdStats has all 9 required fields per spec."""
        from vten.backend.base import CmdStats

        field_names = {f.name for f in fields(CmdStats)}
        expected = {
            "cmd_id", "status", "issue_cycle", "commit_cycle",
            "first_active_cycle", "last_active_cycle",
            "active_cycles", "total_beats", "stall_cycles",
        }
        assert expected.issubset(field_names)

    def test_npu_realistic_axi4_stats(self):
        """Realistic AXI4 BFM stats: DDR burst read for IFM (NPU 3D).

        256-bit bus, 128 beats, typical DDR latency.
        """
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=2, status=3,
            issue_cycle=1000, commit_cycle=1200,
            first_active_cycle=1010, last_active_cycle=1138,
            active_cycles=128, total_beats=128, stall_cycles=1,
        )
        assert s.latency_cycles == 200
        assert s.active_window == 129
        assert s.total_beats == 128
        # Nearly 100% utilization for burst
        assert s.utilization > 0.99

    def test_npu_realistic_axilite_stats(self):
        """Realistic AXI4-Lite stats: single register write (NPU 3D).

        WRITE_REG: very short, ~2-3 cycles.
        """
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=10, status=3,
            issue_cycle=2000, commit_cycle=2003,
            first_active_cycle=2001, last_active_cycle=2002,
            active_cycles=2, total_beats=1, stall_cycles=0,
        )
        assert s.latency_cycles == 3
        assert s.total_beats == 1

    def test_npu_realistic_poll_reg_stats(self):
        """Realistic POLL_REG stats: waiting for LAYER_DONE (NPU 3D).

        Poll at fmapIO ctrl offset 0x054, can take thousands of cycles.
        """
        from vten.backend.base import CmdStats

        s = CmdStats(
            cmd_id=50, status=3,
            issue_cycle=5000, commit_cycle=50000,
            first_active_cycle=5001, last_active_cycle=49999,
            active_cycles=45000, total_beats=45000, stall_cycles=0,
        )
        assert s.latency_cycles == 45000
        assert s.active_window == 44999
