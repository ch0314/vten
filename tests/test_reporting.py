"""Tests for vten/reporting.py — metadata bridge and enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest


# ═══════════════════════════════════════════════════════════════════
# Helpers — lightweight stubs to avoid full pipeline dependency
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StubCommand:
    """Minimal Command stub for testing."""

    op: object
    cmd_id: int
    interface_id: int = 0
    buffer_id: int = 0
    protocol: object = None
    phys_addr: int = 0
    size: int = 0
    dep: list[int] = field(default_factory=list)
    commit_dep: list[int] = field(default_factory=list)
    reg_offset: int = 0
    reg_value: int = 0
    probe: bool = False
    port: str = ""

    def __post_init__(self):
        from vten.spec.models import Protocol
        if self.protocol is None:
            self.protocol = Protocol.AXI4S


@dataclass
class StubCmdStats:
    """Minimal CmdStats stub."""

    cmd_id: int
    status: int = 3
    issue_cycle: int = 0
    commit_cycle: int = 0
    first_active_cycle: int = 0
    last_active_cycle: int = 0
    active_cycles: int = 0
    total_beats: int = 0
    stall_cycles: int = 0

    @property
    def latency_cycles(self) -> int:
        return self.commit_cycle - self.issue_cycle

    @property
    def active_window(self) -> int:
        return self.last_active_cycle - self.first_active_cycle + 1

    @property
    def utilization(self) -> float:
        window = self.active_window
        if window == 0:
            return 0.0
        return self.active_cycles / window

    @property
    def bus_efficiency(self) -> float:
        latency = self.latency_cycles
        if latency == 0:
            return 0.0
        return self.active_cycles / latency


@dataclass
class StubExposedTensor:
    name: str
    origin_path: str
    top_interface: str


@dataclass
class StubInterfaceMapping:
    sub_kernel: str
    sub_interface: str
    top_interface: str | None


@dataclass
class StubFlattenedView:
    exposed_tensors: dict
    interface_mappings: list


@dataclass
class StubCompiledResult:
    commands: list
    buffer_ids: dict
    flattened_view: StubFlattenedView
    iface_id_to_name: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# §1  _status_name
# ═══════════════════════════════════════════════════════════════════


class TestStatusName:

    def test_committed(self):
        from vten.runtime.reporting import _status_name
        assert _status_name(3) == "COMMITTED"

    def test_pending(self):
        from vten.runtime.reporting import _status_name
        assert _status_name(0) == "PENDING"

    def test_error(self):
        from vten.runtime.reporting import _status_name
        assert _status_name(4) == "ERROR"

    def test_unknown(self):
        from vten.runtime.reporting import _status_name
        result = _status_name(99)
        assert "UNKNOWN" in result


# ═══════════════════════════════════════════════════════════════════
# §2  build_command_metadata — unit kernel
# ═══════════════════════════════════════════════════════════════════


class TestBuildCommandMetadataUnit:

    def _make_compiled(self):
        from vten.spec.models import OpCode, Protocol

        commands = [
            StubCommand(op=OpCode.PUSH, cmd_id=0, interface_id=0,
                        buffer_id=0, protocol=Protocol.AXI4S, size=1024),
            StubCommand(op=OpCode.PULL, cmd_id=1, interface_id=0,
                        buffer_id=1, protocol=Protocol.AXI4S, size=2048),
            StubCommand(op=OpCode.WRITE_REG, cmd_id=2, interface_id=1,
                        buffer_id=0, protocol=Protocol.AXI4L,
                        reg_offset=0x10, reg_value=42),
        ]

        view = StubFlattenedView(
            exposed_tensors={
                "data_in": StubExposedTensor("data_in", "_self.data_in", "s_axis"),
                "data_out": StubExposedTensor("data_out", "_self.data_out", "m_axis"),
            },
            interface_mappings=[
                StubInterfaceMapping("_self", "s_axis", "s_axis"),
                StubInterfaceMapping("_self", "m_axis", "m_axis"),
                StubInterfaceMapping("_self", "ctrl", "ctrl"),
            ],
        )

        return StubCompiledResult(
            commands=commands,
            buffer_ids={"data_in": 0, "data_out": 1},
            flattened_view=view,
            iface_id_to_name={0: "s_axis", 1: "ctrl"},
        )

    def test_metadata_count_matches_commands(self):
        from vten.runtime.reporting import build_command_metadata
        compiled = self._make_compiled()
        metadata = build_command_metadata(compiled)
        assert len(metadata) == 3

    def test_push_metadata(self):
        from vten.runtime.reporting import build_command_metadata
        compiled = self._make_compiled()
        metadata = build_command_metadata(compiled)
        push = metadata[0]
        assert push.op_name == "PUSH"
        assert push.interface_name == "s_axis"
        assert push.tensor_name == "data_in"
        assert push.size == 1024

    def test_pull_metadata(self):
        from vten.runtime.reporting import build_command_metadata
        compiled = self._make_compiled()
        metadata = build_command_metadata(compiled)
        pull = metadata[1]
        assert pull.op_name == "PULL"
        assert pull.tensor_name == "data_out"

    def test_write_reg_no_tensor(self):
        from vten.runtime.reporting import build_command_metadata
        compiled = self._make_compiled()
        metadata = build_command_metadata(compiled)
        wreg = metadata[2]
        assert wreg.op_name == "WRITE_REG"
        assert wreg.tensor_name is None
        assert wreg.interface_name == "ctrl"
        assert wreg.reg_offset == 0x10
        assert wreg.reg_value == 42

    def test_sub_kernel_is_self_for_unit(self):
        from vten.runtime.reporting import build_command_metadata
        compiled = self._make_compiled()
        metadata = build_command_metadata(compiled)
        # For unit kernel, sub_kernel derived from origin_path "_self.xxx"
        assert metadata[0].sub_kernel == "_self"


# ═══════════════════════════════════════════════════════════════════
# §3  build_command_metadata — composite kernel
# ═══════════════════════════════════════════════════════════════════


class TestBuildCommandMetadataComposite:

    def test_sub_kernel_populated(self):
        from vten.runtime.reporting import build_command_metadata
        from vten.spec.models import OpCode, Protocol

        commands = [
            StubCommand(op=OpCode.PUSH, cmd_id=0, interface_id=0,
                        buffer_id=0, protocol=Protocol.AXI4S, size=512),
            StubCommand(op=OpCode.WRITE_REG, cmd_id=1, interface_id=1,
                        buffer_id=0, protocol=Protocol.AXI4L),
        ]

        view = StubFlattenedView(
            exposed_tensors={
                "ifm_data": StubExposedTensor("ifm_data", "fmapIO.ifm", "ddr_fmap"),
            },
            interface_mappings=[
                StubInterfaceMapping("fmapIO", "ddr_fmap", "ddr_fmap"),
                StubInterfaceMapping("mac_atu", "ctrl", "ctrl_mac"),
            ],
        )

        compiled = StubCompiledResult(
            commands=commands,
            buffer_ids={"ifm_data": 0},
            flattened_view=view,
            iface_id_to_name={0: "ddr_fmap", 1: "ctrl_mac"},
        )

        metadata = build_command_metadata(compiled)
        assert metadata[0].sub_kernel == "fmapIO"
        assert metadata[0].origin_path == "fmapIO.ifm"
        # WRITE_REG: sub_kernel from interface mapping
        assert metadata[1].sub_kernel == "mac_atu"


# ═══════════════════════════════════════════════════════════════════
# §4  merge_stats_with_metadata
# ═══════════════════════════════════════════════════════════════════


class TestMergeStatsWithMetadata:

    def test_basic_merge(self):
        from vten.runtime.reporting import (
            CommandMetadata,
            merge_stats_with_metadata,
        )

        stats = [
            StubCmdStats(cmd_id=0, status=3, issue_cycle=10, commit_cycle=50,
                         first_active_cycle=15, last_active_cycle=45,
                         active_cycles=20, total_beats=8, stall_cycles=5),
        ]
        metadata = [
            CommandMetadata(
                cmd_id=0, op_name="PUSH", interface_name="s_axis",
                protocol="axi4_stream", tensor_name="data_in",
                size=1024, dep=[], commit_dep=[], sub_kernel="_self",
                origin_path="_self.data_in", port="", probe=False,
                reg_offset=0, reg_value=0,
            ),
        ]

        enriched = merge_stats_with_metadata(stats, metadata)
        assert len(enriched) == 1
        e = enriched[0]
        assert e.op_name == "PUSH"
        assert e.tensor_name == "data_in"
        assert e.status_code == 3
        assert e.status_name == "COMMITTED"
        assert e.latency_cycles == 40
        assert e.active_cycles == 20

    def test_missing_metadata_fallback(self):
        from vten.runtime.reporting import merge_stats_with_metadata

        stats = [StubCmdStats(cmd_id=5, status=3)]
        enriched = merge_stats_with_metadata(stats, [])
        assert len(enriched) == 1
        assert enriched[0].op_name == ""  # fallback

    def test_utilization_carried_through(self):
        from vten.runtime.reporting import (
            CommandMetadata,
            merge_stats_with_metadata,
        )

        stats = [
            StubCmdStats(cmd_id=0, status=3, issue_cycle=0, commit_cycle=100,
                         first_active_cycle=10, last_active_cycle=59,
                         active_cycles=40, total_beats=10, stall_cycles=10),
        ]
        metadata = [
            CommandMetadata(
                cmd_id=0, op_name="PUSH", interface_name="",
                protocol="axi4_stream", tensor_name=None,
                size=0, dep=[], commit_dep=[], sub_kernel=None,
                origin_path=None, port="", probe=False,
                reg_offset=0, reg_value=0,
            ),
        ]

        enriched = merge_stats_with_metadata(stats, metadata)
        assert enriched[0].utilization == 0.8  # 40/50
        assert enriched[0].bus_efficiency == 0.4  # 40/100


# ═══════════════════════════════════════════════════════════════════
# §5  EnrichedCmdStats.to_dict
# ═══════════════════════════════════════════════════════════════════


class TestEnrichedCmdStatsToDict:

    def test_to_dict_basic_fields(self):
        from vten.runtime.reporting import EnrichedCmdStats

        e = EnrichedCmdStats(
            cmd_id=0, op_name="PUSH", interface_name="s_axis",
            protocol="axi4_stream", tensor_name="data_in",
            size=1024, dep=[], commit_dep=[], sub_kernel="_self",
            origin_path="_self.data_in", port="", probe=False,
            reg_offset=0, reg_value=0,
            status_code=3, status_name="COMMITTED",
            issue_cycle=10, commit_cycle=50,
            active_cycles=20, stall_cycles=5,
            total_beats=8, latency_cycles=40,
            utilization=0.8, bus_efficiency=0.5,
        )
        d = e.to_dict()
        assert d["op"] == "PUSH"
        assert d["status"] == 3
        assert d["status_name"] == "COMMITTED"
        assert d["tensor"] == "data_in"
        assert d["size"] == 1024

    def test_to_dict_no_tensor_for_barrier(self):
        from vten.runtime.reporting import EnrichedCmdStats

        e = EnrichedCmdStats(
            cmd_id=0, op_name="BARRIER", interface_name="",
            protocol="", tensor_name=None,
            size=0, dep=[], commit_dep=[], sub_kernel=None,
            origin_path=None, port="", probe=False,
            reg_offset=0, reg_value=0,
            status_code=3, status_name="COMMITTED",
            issue_cycle=0, commit_cycle=0,
            active_cycles=0, stall_cycles=0,
            total_beats=0, latency_cycles=0,
            utilization=0.0, bus_efficiency=0.0,
        )
        d = e.to_dict()
        assert "tensor" not in d

    def test_to_dict_sub_kernel_only_for_composite(self):
        from vten.runtime.reporting import EnrichedCmdStats

        # _self should not appear in output
        e = EnrichedCmdStats(
            cmd_id=0, op_name="PUSH", interface_name="",
            protocol="", tensor_name=None,
            size=0, dep=[], commit_dep=[], sub_kernel="_self",
            origin_path="_self.x", port="", probe=False,
            reg_offset=0, reg_value=0,
            status_code=3, status_name="COMMITTED",
            issue_cycle=0, commit_cycle=0,
            active_cycles=0, stall_cycles=0,
            total_beats=0, latency_cycles=0,
            utilization=0.0, bus_efficiency=0.0,
        )
        d = e.to_dict()
        assert "sub_kernel" not in d

    def test_to_dict_reg_fields_for_write_reg(self):
        from vten.runtime.reporting import EnrichedCmdStats

        e = EnrichedCmdStats(
            cmd_id=0, op_name="WRITE_REG", interface_name="ctrl",
            protocol="axi4_lite", tensor_name=None,
            size=0, dep=[], commit_dep=[], sub_kernel=None,
            origin_path=None, port="", probe=False,
            reg_offset=0x10, reg_value=42,
            status_code=3, status_name="COMMITTED",
            issue_cycle=0, commit_cycle=0,
            active_cycles=0, stall_cycles=0,
            total_beats=0, latency_cycles=0,
            utilization=0.0, bus_efficiency=0.0,
        )
        d = e.to_dict()
        assert d["reg_offset"] == 0x10
        assert d["reg_value"] == 42

    def test_to_dict_probe_flag(self):
        from vten.runtime.reporting import EnrichedCmdStats

        e = EnrichedCmdStats(
            cmd_id=0, op_name="PUSH", interface_name="",
            protocol="", tensor_name="x",
            size=0, dep=[], commit_dep=[], sub_kernel=None,
            origin_path=None, port="", probe=True,
            reg_offset=0, reg_value=0,
            status_code=3, status_name="COMMITTED",
            issue_cycle=0, commit_cycle=0,
            active_cycles=0, stall_cycles=0,
            total_beats=0, latency_cycles=0,
            utilization=0.0, bus_efficiency=0.0,
        )
        d = e.to_dict()
        assert d["probe"] is True

    def test_to_dict_port_field(self):
        from vten.runtime.reporting import EnrichedCmdStats

        e = EnrichedCmdStats(
            cmd_id=0, op_name="PUSH", interface_name="",
            protocol="", tensor_name="x",
            size=0, dep=[], commit_dep=[], sub_kernel=None,
            origin_path=None, port="port_a", probe=False,
            reg_offset=0, reg_value=0,
            status_code=3, status_name="COMMITTED",
            issue_cycle=0, commit_cycle=0,
            active_cycles=0, stall_cycles=0,
            total_beats=0, latency_cycles=0,
            utilization=0.0, bus_efficiency=0.0,
        )
        d = e.to_dict()
        assert d["port"] == "port_a"


# ═══════════════════════════════════════════════════════════════════
# §6  VerificationResult / ProbeResult dataclasses
# ═══════════════════════════════════════════════════════════════════


class TestVerificationResult:

    def test_pass_result(self):
        from vten.runtime.reporting import VerificationResult
        vr = VerificationResult(tensor_name="out", passed=True)
        assert vr.passed
        assert vr.max_diff == 0.0

    def test_fail_result(self):
        from vten.runtime.reporting import VerificationResult
        vr = VerificationResult(
            tensor_name="out", passed=False,
            max_diff=0.5, shape=(32, 32),
        )
        assert not vr.passed
        assert vr.max_diff == 0.5
        assert vr.shape == (32, 32)


class TestProbeResult:

    def test_probe_match(self):
        from vten.runtime.reporting import ProbeResult
        pr = ProbeResult(
            probe_point="mac_atu.ifm_in",
            connection="fmapIO.ifm_out -> mac_atu.ifm_in",
            passed=True,
        )
        assert pr.passed

    def test_probe_mismatch(self):
        from vten.runtime.reporting import ProbeResult
        pr = ProbeResult(
            probe_point="mac_atu.ifm_in",
            connection="fmapIO.ifm_out -> mac_atu.ifm_in",
            passed=False,
            max_diff=0.03,
            mismatch_count=5,
            first_mismatch_index=[0, 14],
            expected_value=1.5,
            actual_value=1.53,
        )
        assert not pr.passed
        assert pr.mismatch_count == 5


# ═══════════════════════════════════════════════════════════════════
# §7  Report terminal format — enriched data
# ═══════════════════════════════════════════════════════════════════


class TestReportTerminalEnriched:

    def _setup_enriched_result(self, tmp_path, *, sub_kernel=None):
        import json

        kernel_name = "test_kernel"
        test_name = "TestFoo"
        results = tmp_path / "results" / kernel_name / test_name
        results.mkdir(parents=True)

        (results / "summary.json").write_text(json.dumps({
            "test_name": test_name,
            "kernel": kernel_name,
            "status": "PASS",
            "total_cycles": 500,
            "configs_run": 1,
            "configs_passed": 1,
            "verification_count": 1,
            "verification_passed": 1,
            "verification_results": [
                {"tensor": "data_out", "passed": True, "max_diff": 0.0},
            ],
        }))

        cmd = {
            "cmd_id": 0,
            "op": "PUSH",
            "interface": "s_axis",
            "protocol": "axi4_stream",
            "tensor": "data_in",
            "size": 1024,
            "status": 3,
            "status_name": "COMMITTED",
            "issue_cycle": 10,
            "commit_cycle": 50,
            "latency_cycles": 40,
            "active_cycles": 20,
            "stall_cycles": 0,
            "total_beats": 8,
            "utilization": 0.8,
            "bus_efficiency": 0.5,
        }
        if sub_kernel:
            cmd["sub_kernel"] = sub_kernel

        (results / "stats.json").write_text(json.dumps({"commands": [cmd]}))
        return tmp_path

    def test_terminal_shows_op_name(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "PUSH" in report

    def test_terminal_shows_interface(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "s_axis" in report

    def test_terminal_shows_tensor(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "data_in" in report

    def test_terminal_shows_status_name(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "COMMITTED" in report

    def test_terminal_shows_verification(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="terminal")
        assert "data_out" in report
        assert "PASS" in report

    def test_terminal_sub_kernel_grouping(self, tmp_path):
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path, sub_kernel="fmapIO")
        report = generate_report(str(project), format="terminal")
        assert "[fmapIO]" in report

    def test_nested_results_dir(self, tmp_path):
        """Verify nested results/<kernel>/<test>/ layout is scanned."""
        from vten.cli.report import generate_report

        project = self._setup_enriched_result(tmp_path)
        report = generate_report(str(project), format="json")
        import json
        parsed = json.loads(report)
        assert len(parsed) >= 1
        assert "test_kernel" in parsed[0]["test_name"]


# ═══════════════════════════════════════════════════════════════════
# §8  Report terminal format — legacy data (backward compat)
# ═══════════════════════════════════════════════════════════════════


class TestReportTerminalLegacy:

    def test_legacy_status_int_resolved(self, tmp_path):
        """Legacy stats.json with status as int should still work."""
        import json
        from vten.cli.report import generate_report

        results = tmp_path / "results" / "test_old"
        results.mkdir(parents=True)

        (results / "summary.json").write_text(json.dumps({
            "test_name": "test_old", "status": "PASS",
            "total_cycles": 100, "configs_run": 1, "configs_passed": 1,
        }))
        (results / "stats.json").write_text(json.dumps({
            "commands": [
                {"cmd_id": 0, "status": 3, "issue_cycle": 0, "commit_cycle": 50,
                 "active_cycles": 10, "stall_cycles": 0, "total_beats": 5},
            ],
        }))

        report = generate_report(str(tmp_path), format="terminal")
        # Should resolve status 3 → COMMITTED
        assert "COMMITTED" in report
