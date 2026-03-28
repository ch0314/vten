"""Tests for probe verification pipeline (Phase 2-5).

Covers:
  §1 Probe golden serialization (engine._serialize_probe_golden)
  §2 Probe buffer map building (engine._build_probe_buffer_map)
  §3 ProbeMismatchError attributes and raise_backend_error mapping
  §4 Mismatches JSONL parsing (sim_base._parse_mismatch_file)
  §5 Waveform TCL generation (codegen)
  §6 Probe error wiring in generated testbench
  §7 Controller probe_error port
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest
import torch

from vten.errors import BackendError, ProbeMismatchError


# ═══════════════════════════════════════════════════════════════════
# §1  Probe golden serialization
# ═══════════════════════════════════════════════════════════════════


class TestProbeGoldenSerialization:
    """Engine._serialize_probe_golden populates ProbePoint.serialized_golden."""

    def _make_view_with_probe(self):
        """Create a FlattenedKernelView with one probe point."""
        from vten.runtime.flattener import (
            FlattenedKernelView,
            KernelInstance,
            ProbePoint,
        )
        from vten.spec.models import (
            InterfaceSpec,
            KernelSpec,
            PackingScheme,
            Protocol,
        )

        spec = KernelSpec(
            kernel_name="scale",
            rtl_top="rtl/scale.sv",
            interfaces={
                "data_out": InterfaceSpec(
                    name="data_out",
                    rtl_port="m_axis_out",
                    protocol=Protocol.AXI4S,
                    tensor="data_out",
                    packing=PackingScheme(element_width=8, elements_per_beat=32),
                ),
            },
        )

        ki = KernelInstance(
            name="scale",
            spec=spec,
            kernel_class=type("DummyKernel", (), {}),
            kernel_class_instance=None,
            runtime_params={},
        )

        # Create a connection-like object
        class MockConnection:
            source_sub = "scale"
            source_name = "data_out"
            source_interface = "data_out"

        probe = ProbePoint(connection=MockConnection())
        view = FlattenedKernelView(
            name="test_composite",
            top_spec=spec,
            sub_kernels={"scale": ki},
            interface_mappings=[],
            exposed_tensors={},
            probe_points=[probe],
            connections=[],
        )
        return view, probe

    def test_serialization_populates_golden_data(self):
        """Golden tensor is serialized and stored on ProbePoint."""
        from vten.runtime.engine import RuntimeEngine

        view, probe = self._make_view_with_probe()
        golden = torch.arange(32, dtype=torch.int8)
        internal_golden = {("scale", "data_out"): golden}

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        engine._serialize_probe_golden(view, internal_golden)

        assert probe.serialized_golden is not None
        assert len(probe.serialized_golden) == 32  # 32 int8 elements = 32 bytes

    def test_serialization_matches_raw_bytes(self):
        """Serialized bytes match the raw tensor data."""
        from vten.runtime.engine import RuntimeEngine

        view, probe = self._make_view_with_probe()
        golden = torch.tensor([1, 2, -1, 127, -128] + [0] * 27, dtype=torch.int8)
        internal_golden = {("scale", "data_out"): golden}

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        engine._serialize_probe_golden(view, internal_golden)

        raw = probe.serialized_golden
        assert raw[0] == 1
        assert raw[1] == 2
        assert raw[2] == 0xFF  # -1 in unsigned byte
        assert raw[3] == 127
        assert raw[4] == 0x80  # -128 in unsigned byte

    def test_no_golden_leaves_probe_unchanged(self):
        """Missing golden key → ProbePoint.serialized_golden stays None."""
        from vten.runtime.engine import RuntimeEngine

        view, probe = self._make_view_with_probe()
        internal_golden = {("other_sub", "other_port"): torch.zeros(32, dtype=torch.int8)}

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        engine._serialize_probe_golden(view, internal_golden)

        assert probe.serialized_golden is None

    def test_none_golden_is_noop(self):
        """None internal_probe_golden → no-op, no error."""
        from vten.runtime.engine import RuntimeEngine

        view, probe = self._make_view_with_probe()

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        engine._serialize_probe_golden(view, None)

        assert probe.serialized_golden is None


# ═══════════════════════════════════════════════════════════════════
# §2  Probe buffer map building
# ═══════════════════════════════════════════════════════════════════


class TestProbeBufferMap:
    """Engine._build_probe_buffer_map creates probe_index → buffer_id mapping."""

    @staticmethod
    def _make_view(probe_points):
        from vten.runtime.flattener import FlattenedKernelView
        from vten.spec.models import KernelSpec

        return FlattenedKernelView(
            name="test",
            top_spec=KernelSpec(kernel_name="test", rtl_top="test.sv"),
            sub_kernels={},
            interface_mappings=[],
            exposed_tensors={},
            probe_points=probe_points,
            connections=[],
        )

    def test_single_probe_with_buffer(self):
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import ProbePoint

        class MockConn:
            source_sub = "sub"
            source_name = "out"
            source_interface = "out"

        probe = ProbePoint(connection=MockConn())
        probe.golden_buffer_id = 5

        mapping = RuntimeEngine._build_probe_buffer_map(self._make_view([probe]))
        assert mapping == {0: 5}

    def test_multiple_probes(self):
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import ProbePoint

        class MockConn:
            source_sub = "sub"
            source_name = "out"
            source_interface = "out"

        p0 = ProbePoint(connection=MockConn())
        p0.golden_buffer_id = 2
        p1 = ProbePoint(connection=MockConn())
        p1.golden_buffer_id = 7

        mapping = RuntimeEngine._build_probe_buffer_map(self._make_view([p0, p1]))
        assert mapping == {0: 2, 1: 7}

    def test_probe_without_buffer_skipped(self):
        """Probes without golden_buffer_id are not included."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import ProbePoint

        class MockConn:
            source_sub = "sub"
            source_name = "out"
            source_interface = "out"

        p0 = ProbePoint(connection=MockConn())
        p0.golden_buffer_id = None

        mapping = RuntimeEngine._build_probe_buffer_map(self._make_view([p0]))
        assert mapping == {}

    def test_empty_probes(self):
        from vten.runtime.engine import RuntimeEngine

        mapping = RuntimeEngine._build_probe_buffer_map(self._make_view([]))
        assert mapping == {}


# ═══════════════════════════════════════════════════════════════════
# §3  ProbeMismatchError
# ═══════════════════════════════════════════════════════════════════


class TestProbeMismatchError:
    """ProbeMismatchError attributes and raise_backend_error mapping."""

    def test_inherits_backend_error(self):
        assert issubclass(ProbeMismatchError, BackendError)

    def test_default_attributes(self):
        e = ProbeMismatchError("mismatch")
        assert e.cmd_id == 0
        assert e.beat_index == 0
        assert e.mismatches == []

    def test_custom_attributes(self):
        mismatches = [{"beat": 5, "cycle": 100}]
        e = ProbeMismatchError(
            "probe fail",
            cmd_id=3,
            beat_index=5,
            mismatches=mismatches,
        )
        assert e.cmd_id == 3
        assert e.beat_index == 5
        assert len(e.mismatches) == 1
        assert e.mismatches[0]["beat"] == 5

    def test_raise_backend_error_code_8(self):
        """Error code 8 maps to ProbeMismatchError."""
        from vten.backend.base import raise_backend_error

        with pytest.raises(ProbeMismatchError) as exc_info:
            raise_backend_error(code=8, cmd_id=4, message="probe mismatch")
        assert exc_info.value.cmd_id == 4

    def test_raise_backend_error_includes_context(self):
        from vten.backend.base import raise_backend_error

        with pytest.raises(ProbeMismatchError) as exc_info:
            raise_backend_error(code=8, cmd_id=2, message="mismatch on wire")
        assert exc_info.value.context["error_code"] == 8
        assert exc_info.value.context["cmd_id"] == 2


# ═══════════════════════════════════════════════════════════════════
# §4  Mismatches JSONL parsing
# ═══════════════════════════════════════════════════════════════════


class TestMismatchFileParsing:
    """SimBackend._parse_mismatch_file reads mismatches.jsonl."""

    def _make_backend(self, mismatch_dir):
        """Create a minimal SimBackend-like object with _parse_mismatch_file."""
        from vten.backend.sim_base import SimBackend

        # SimBackend is abstract, so we instantiate minimally
        class _Stub(SimBackend):
            def _start_simulator(self):
                pass

        config = {"_mismatch_dir": str(mismatch_dir)}
        # Bypass __init__ and set config directly
        obj = object.__new__(_Stub)
        obj._config = config
        return obj

    def test_parse_single_mismatch(self, tmp_path):
        entry = {"cmd_id": 0, "cycle": 46, "beat": 0,
                 "expected_hi": "0x00000000", "expected_lo": "0x00000000",
                 "actual_hi": "0x042ECEC6", "actual_lo": "0x80B866CC"}
        (tmp_path / "mismatches.jsonl").write_text(json.dumps(entry) + "\n")

        backend = self._make_backend(tmp_path)
        result = backend._parse_mismatch_file()

        assert len(result) == 1
        assert result[0]["beat"] == 0
        assert result[0]["cycle"] == 46
        assert result[0]["actual_hi"] == "0x042ECEC6"

    def test_parse_multiple_mismatches(self, tmp_path):
        lines = [
            json.dumps({"cmd_id": 0, "cycle": 46, "beat": 0,
                        "expected_hi": "0x0", "expected_lo": "0x0",
                        "actual_hi": "0x1", "actual_lo": "0x2"}),
            json.dumps({"cmd_id": 0, "cycle": 47, "beat": 1,
                        "expected_hi": "0x0", "expected_lo": "0x0",
                        "actual_hi": "0x3", "actual_lo": "0x4"}),
        ]
        (tmp_path / "mismatches.jsonl").write_text("\n".join(lines) + "\n")

        backend = self._make_backend(tmp_path)
        result = backend._parse_mismatch_file()

        assert len(result) == 2
        assert result[1]["beat"] == 1

    def test_missing_file_returns_empty(self, tmp_path):
        backend = self._make_backend(tmp_path)
        result = backend._parse_mismatch_file()
        assert result == []

    def test_no_mismatch_dir_returns_empty(self):
        from vten.backend.sim_base import SimBackend

        class _Stub(SimBackend):
            def _start_simulator(self):
                pass

        obj = object.__new__(_Stub)
        obj._config = {}
        result = obj._parse_mismatch_file()
        assert result == []

    def test_malformed_json_returns_empty(self, tmp_path):
        (tmp_path / "mismatches.jsonl").write_text("not json\n")

        backend = self._make_backend(tmp_path)
        result = backend._parse_mismatch_file()
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# §5  Waveform TCL generation
# ═══════════════════════════════════════════════════════════════════


class TestWaveformTclGeneration:
    """SVGenerator.generate() produces waveform.tcl from template."""

    def _make_generator(self, probe_bfms=None):
        from vten.codegen.sv_generator import SVGenerator
        from vten.runtime.ir import BFMConfig
        from vten.spec.models import (
            InterfaceSpec,
            KernelSpec,
            PackingScheme,
            Protocol,
        )

        spec = KernelSpec(
            kernel_name="passthrough",
            rtl_top="rtl/passthrough.sv",
            interfaces={
                "in": InterfaceSpec(
                    name="in",
                    rtl_port="s_axis_in",
                    protocol=Protocol.AXI4S,
                    tensor="data_in",
                    packing=PackingScheme(element_width=8, elements_per_beat=4),
                ),
                "out": InterfaceSpec(
                    name="out",
                    rtl_port="m_axis_out",
                    protocol=Protocol.AXI4S,
                    tensor="data_out",
                    packing=PackingScheme(element_width=8, elements_per_beat=4),
                ),
            },
        )
        bfm_configs = [
            BFMConfig(
                interface_name="in",
                protocol=Protocol.AXI4S,
                data_width=32,
                role="master",
            ),
            BFMConfig(
                interface_name="out",
                protocol=Protocol.AXI4S,
                data_width=32,
                role="slave",
            ),
        ]
        config = {
            "project": {"name": "test", "version": "0.1.0"},
            "rtl": {"sources": ["rtl/passthrough.sv"], "top_module": "passthrough"},
            "backend": {"xsim": {"vivado_path": "/tools/Xilinx/Vivado/2023.2"}},
        }
        return SVGenerator(spec, bfm_configs, config, probe_bfms=probe_bfms)

    def test_waveform_tcl_generated(self, tmp_path):
        gen = self._make_generator()
        gen.generate(str(tmp_path))
        assert (tmp_path / "waveform.tcl").exists()

    def test_waveform_tcl_logs_dut(self, tmp_path):
        gen = self._make_generator()
        gen.generate(str(tmp_path))
        content = (tmp_path / "waveform.tcl").read_text()
        assert "log_wave -recursive /tb_top/dut/*" in content

    def test_waveform_tcl_logs_bfms(self, tmp_path):
        gen = self._make_generator()
        gen.generate(str(tmp_path))
        content = (tmp_path / "waveform.tcl").read_text()
        assert "bfm_in" in content
        assert "bfm_out" in content

    def test_waveform_tcl_logs_infrastructure(self, tmp_path):
        gen = self._make_generator()
        gen.generate(str(tmp_path))
        content = (tmp_path / "waveform.tcl").read_text()
        assert "shm_ctrl" in content
        assert "scheduler" in content

    def test_waveform_tcl_run_all(self, tmp_path):
        gen = self._make_generator()
        gen.generate(str(tmp_path))
        content = (tmp_path / "waveform.tcl").read_text()
        assert "run all" in content

    def test_waveform_tcl_with_probe_bfms(self, tmp_path):
        probes = [{"probe_index": 0, "data_width": 256, "wire_name": "internal_0"}]
        gen = self._make_generator(probe_bfms=probes)
        gen.generate(str(tmp_path))
        content = (tmp_path / "waveform.tcl").read_text()
        assert "probe_0" in content


# ═══════════════════════════════════════════════════════════════════
# §6  Probe error wiring in generated testbench
# ═══════════════════════════════════════════════════════════════════


class TestProbeErrorWiring:
    """tb_top.sv.j2 correctly wires probe_error to controller."""

    def _generate_tb(self, tmp_path, probe_bfms=None):
        from vten.codegen.sv_generator import SVGenerator
        from vten.runtime.ir import BFMConfig
        from vten.spec.models import (
            InterfaceSpec,
            KernelSpec,
            PackingScheme,
            Protocol,
        )

        spec = KernelSpec(
            kernel_name="pt",
            rtl_top="rtl/pt.sv",
            interfaces={
                "in": InterfaceSpec(
                    name="in",
                    rtl_port="s_axis_in",
                    protocol=Protocol.AXI4S,
                    tensor="data_in",
                    packing=PackingScheme(element_width=8, elements_per_beat=4),
                ),
            },
        )
        bfm_configs = [
            BFMConfig(
                interface_name="in",
                protocol=Protocol.AXI4S,
                data_width=32,
                role="master",
            ),
        ]
        config = {
            "project": {"name": "test", "version": "0.1.0"},
            "rtl": {"sources": ["rtl/pt.sv"], "top_module": "pt"},
            "backend": {"xsim": {"vivado_path": ""}},
        }
        gen = SVGenerator(spec, bfm_configs, config, probe_bfms=probe_bfms)
        gen.generate(str(tmp_path))
        return (tmp_path / "tb_top.sv").read_text()

    def test_no_probes_ties_to_zero(self, tmp_path):
        """Without probes, controller.probe_error is tied to 1'b0."""
        content = self._generate_tb(tmp_path)
        assert ".probe_error(1'b0)" in content

    def test_single_probe_wires_error(self, tmp_path):
        """Single probe: probe_error_0 → probe_error_any → controller."""
        probes = [{"probe_index": 0, "data_width": 256, "wire_name": "internal_0"}]
        content = self._generate_tb(tmp_path, probe_bfms=probes)
        assert "logic probe_error_0" in content
        assert "logic probe_error_any" in content
        assert "assign probe_error_any = probe_error_0" in content
        assert ".probe_error(probe_error_any)" in content

    def test_multiple_probes_or_together(self, tmp_path):
        """Multiple probes: OR'd into probe_error_any."""
        probes = [
            {"probe_index": 0, "data_width": 256, "wire_name": "internal_0"},
            {"probe_index": 1, "data_width": 128, "wire_name": "internal_1"},
        ]
        content = self._generate_tb(tmp_path, probe_bfms=probes)
        assert "logic probe_error_0" in content
        assert "logic probe_error_1" in content
        assert "probe_error_0 | probe_error_1" in content

    def test_probe_bfm_has_error_output(self, tmp_path):
        """Probe BFM instance connects .probe_error output."""
        probes = [{"probe_index": 0, "data_width": 256, "wire_name": "internal_0"}]
        content = self._generate_tb(tmp_path, probe_bfms=probes)
        assert ".probe_error(probe_error_0)" in content


# ═══════════════════════════════════════════════════════════════════
# §7  Controller probe_error port
# ═══════════════════════════════════════════════════════════════════


class TestControllerProbeError:
    """vten_shm_controller.sv has probe_error input port."""

    @pytest.fixture(autouse=True)
    def _load_sv(self):
        sv_path = Path(__file__).resolve().parent.parent / "vten_sv" / "vten_shm_controller.sv"
        self.text = sv_path.read_text()

    def test_probe_error_port_declared(self):
        assert re.search(r"input\s+logic\s+probe_error", self.text)

    def test_execute_checks_probe_error(self):
        """S_EXECUTE state checks probe_error alongside sched_error."""
        assert "sched_error || probe_error" in self.text

    def test_error_state_handles_probe(self):
        """S_ERROR sends ERR_PROBE_MISMATCH (8) for probe errors."""
        assert "probe_error && !sched_error" in self.text

    def test_drain_checks_probe_error(self):
        """S_DRAIN also checks for probe errors."""
        # Find the S_DRAIN case block
        drain_match = re.search(
            r"S_DRAIN:\s*begin(.*?)end\b", self.text, re.DOTALL
        )
        assert drain_match is not None
        drain_body = drain_match.group(1)
        assert "probe_error" in drain_body


class TestProbeBfmSV:
    """vten_bfm_probe.sv has probe_error output and $stop support."""

    @pytest.fixture(autouse=True)
    def _load_sv(self):
        sv_path = Path(__file__).resolve().parent.parent / "vten_sv" / "vten_bfm_probe.sv"
        self.text = sv_path.read_text()

    def test_probe_error_output_declared(self):
        assert re.search(r"output\s+logic\s+probe_error", self.text)

    def test_probe_error_set_on_mismatch(self):
        """On mismatch, probe_error is asserted."""
        assert "probe_error <= 1'b1" in self.text

    def test_stop_on_gui_flag(self):
        """$stop invoked when FLAG_PAUSE_ON_MISMATCH (0x08) is set."""
        assert "$stop" in self.text
        assert "vten_read_flags()" in self.text

    def test_plusargs_resolution(self):
        """Buffer ID resolved via $value$plusargs."""
        assert "$value$plusargs" in self.text
        assert "PROBE_GOLDEN_" in self.text

    def test_error_stops_comparison(self):
        """After first mismatch, further comparison is skipped."""
        assert "!probe_error" in self.text
