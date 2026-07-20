"""Tests for InferenceModel — whole-network orchestration + graph capture.

Builds a small fan-out model on the CPU backend (kernel forward() as reference
DUT, no FPGA) using the scale_add example kernels (ScaleKernel, OffsetKernel)
and asserts:

  (a) output VALUES are correct vs a hand-computed torch reference, and
  (b) the captured GRAPH has the expected nodes and edges, including the
      fan-out (one tensor consumed by two nodes).

Matches the sys.path / _spec pattern used by tests/test_inference.py so no
chdir is required (each kernel's spec is parsed explicitly and passed via
``_spec=``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from vten import InferenceModel, InferenceSession
from vten.inference import GraphNode, _Node

# ── Make the scale_add example kernels importable ──

_SCALE_ADD_DIR = Path(__file__).resolve().parent.parent / "examples" / "scale_add"
for _p in (
    str(_SCALE_ADD_DIR / "kernels" / "scale"),
    str(_SCALE_ADD_DIR / "kernels" / "offset"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scale_kernel import ScaleKernel   # noqa: E402
from offset_kernel import OffsetKernel  # noqa: E402

from vten.spec.parser import parse_kernel_spec  # noqa: E402

N = 32
SCALE_FACTOR = 2
OFF1 = 1
OFF2 = 2


@pytest.fixture(scope="module")
def specs():
    return {
        "scale": parse_kernel_spec(_SCALE_ADD_DIR / "kernels" / "scale" / "kernel_spec.yaml"),
        "offset": parse_kernel_spec(_SCALE_ADD_DIR / "kernels" / "offset" / "kernel_spec.yaml"),
    }


@pytest.fixture
def session():
    return InferenceSession("cpu", project_dir=str(_SCALE_ADD_DIR), log_level="WARNING")


class FanOutNet(InferenceModel):
    """h = scale(x); a = off1(h); b = off2(h)  — h fans out to off1 and off2."""

    def __init__(self, session, specs):
        super().__init__(session)
        self._specs = specs

    def build(self):
        self.scale = self.stage(
            ScaleKernel, scale_factor=SCALE_FACTOR, N=N,
            input_name="data_in", output_name="data_out",
            name="scale", _spec=self._specs["scale"],
        )
        self.off1 = self.stage(
            OffsetKernel, offset_value=OFF1, N=N,
            input_name="data_in", output_name="data_out",
            name="off1", _spec=self._specs["offset"],
        )
        self.off2 = self.stage(
            OffsetKernel, offset_value=OFF2, N=N,
            input_name="data_in", output_name="data_out",
            name="off2", _spec=self._specs["offset"],
        )

    def forward(self, x):
        h = self.scale(x)
        a = self.off1(h)
        b = self.off2(h)
        return a, b


def _reference(x: torch.Tensor):
    h = (x.to(torch.int16) * SCALE_FACTOR).clamp(-128, 127)
    a = (h + OFF1).clamp(-128, 127).to(torch.int8)
    b = (h + OFF2).clamp(-128, 127).to(torch.int8)
    return a, b


# ── (a) Output correctness ──


class TestOutputValues:

    def test_outputs_match_torch_reference(self, session, specs):
        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        a_t, b_t = net(x)
        a, b = a_t.cpu(), b_t.cpu()
        ref_a, ref_b = _reference(x)
        assert torch.equal(a, ref_a)
        assert torch.equal(b, ref_b)

    def test_outputs_differ_between_branches(self, session, specs):
        """off1 (+1) and off2 (+2) diverge — they are genuinely two nodes."""
        net = FanOutNet(session, specs)
        x = torch.zeros(N, dtype=torch.int8)
        a_t, b_t = net(x)
        # h == 0, so a == 1 everywhere, b == 2 everywhere.
        assert torch.equal(a_t.cpu(), torch.ones(N, dtype=torch.int8))
        assert torch.equal(b_t.cpu(), torch.full((N,), 2, dtype=torch.int8))

    def test_rerun_resets_graph(self, session, specs):
        """A second forward() starts a fresh trace (no accumulation)."""
        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        net(x)
        first = net.graph()
        net(x)
        second = net.graph()
        assert len(second["nodes"]) == len(first["nodes"]) == 3


# ── (b) Graph capture: nodes + edges + fan-out ──


class TestGraphCapture:

    def _built_graph(self, session, specs):
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))
        return net.graph()

    def test_expected_nodes(self, session, specs):
        g = self._built_graph(session, specs)
        names = [n["name"] for n in g["nodes"]]
        assert names == ["scale", "off1", "off2"]  # execution order
        kernels = {n["name"]: n["kernel"] for n in g["nodes"]}
        assert kernels == {
            "scale": "ScaleKernel",
            "off1": "OffsetKernel",
            "off2": "OffsetKernel",
        }

    def test_params_captured(self, session, specs):
        g = self._built_graph(session, specs)
        params = {n["name"]: n["params"] for n in g["nodes"]}
        assert params["scale"]["scale_factor"] == SCALE_FACTOR
        assert params["off1"]["offset_value"] == OFF1
        assert params["off2"]["offset_value"] == OFF2

    def test_graph_input_edge(self, session, specs):
        """x entered from outside → 'scale' has a source == 'input' edge."""
        g = self._built_graph(session, specs)
        scale_inputs = next(n for n in g["nodes"] if n["name"] == "scale")["inputs"]
        assert scale_inputs == [{"tensor_name": "data_in", "source": "input"}]

    def test_edges_producer_to_consumer(self, session, specs):
        g = self._built_graph(session, specs)
        edges = {(e["src"], e["dst"]) for e in g["edges"]}
        assert ("input", "scale") in edges
        assert ("scale", "off1") in edges
        assert ("scale", "off2") in edges

    def test_fan_out_scale_consumed_by_two_nodes(self, session, specs):
        """THE key assertion: h (scale's output) fans out to off1 AND off2."""
        g = self._built_graph(session, specs)
        consumers = sorted(e["dst"] for e in g["edges"] if e["src"] == "scale")
        assert consumers == ["off1", "off2"]

    def test_edge_tensor_names(self, session, specs):
        g = self._built_graph(session, specs)
        for e in g["edges"]:
            assert e["tensor_name"] == "data_in"

    def test_graph_is_json_serializable(self, session, specs):
        import json

        g = self._built_graph(session, specs)
        # Must round-trip through JSON for the later verify/perf agent.
        assert json.loads(json.dumps(g)) == g


# ── API surface / hook points (for the later verify/perf agent) ──


class TestApiSurface:

    def test_stage_auto_names_are_unique(self, session, specs):
        class AutoNet(InferenceModel):
            def build(self):
                self.a = self.stage(OffsetKernel, offset_value=1, N=N,
                                    _spec=specs["offset"])
                self.b = self.stage(OffsetKernel, offset_value=2, N=N,
                                    _spec=specs["offset"])

            def forward(self, x):
                return self.b(self.a(x))

        net = AutoNet(session)
        net(torch.zeros(N, dtype=torch.int8))
        names = [n["name"] for n in net.graph()["nodes"]]
        assert len(names) == len(set(names))  # no collisions
        assert names[0] == "offset"

    def test_duplicate_explicit_name_raises(self, session, specs):
        class DupNet(InferenceModel):
            def build(self):
                self.stage(OffsetKernel, offset_value=1, N=N, name="dup",
                          _spec=specs["offset"])
                self.stage(OffsetKernel, offset_value=2, N=N, name="dup",
                          _spec=specs["offset"])

            def forward(self, x):
                return x

        net = DupNet(session)
        with pytest.raises(ValueError, match="duplicate stage name"):
            net(torch.zeros(N, dtype=torch.int8))

    def test_build_called_once(self, session, specs):
        net = FanOutNet(session, specs)
        calls = {"n": 0}
        orig_build = net.build

        def counting_build():
            calls["n"] += 1
            orig_build()

        net.build = counting_build
        x = torch.zeros(N, dtype=torch.int8)
        net(x)
        net(x)
        assert calls["n"] == 1

    def test_hook_points_are_invoked(self, session, specs):
        """_on_before_node / _on_after_node fire per node (attach points)."""
        seen = {"before": [], "after": []}

        class HookNet(FanOutNet):
            def _on_before_node(self, node, inputs):
                assert isinstance(node, _Node)
                seen["before"].append(node.name)

            def _on_after_node(self, node, record, output):
                assert isinstance(record, GraphNode)
                seen["after"].append(record.node_name)

        net = HookNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))
        assert seen["before"] == ["scale", "off1", "off2"]
        assert seen["after"] == ["scale", "off1", "off2"]

    def test_graphnode_hook_slots_on_plain_run(self, session, specs):
        """Plain (non-verify) run: verification empty; stats None on cpu.

        The post-node hook fires on every run. Without verify=True there is no
        verification outcome to record (slot stays empty). The cpu backend
        emits no CmdStats, so the perf slot is set to None (graceful degrade),
        never a crash.
        """
        net = FanOutNet(session, specs)
        net(torch.zeros(N, dtype=torch.int8))
        for rec in net._graph:
            assert rec.verification == {}
            assert rec.stats is None

    def test_forward_not_implemented_by_base(self, session):
        net = InferenceModel(session)
        with pytest.raises(NotImplementedError):
            net(torch.zeros(N, dtype=torch.int8))


# ── Slice C: per-node + end-to-end verification ──


class TestPerNodeVerification:

    def test_graphnode_verification_populated_on_verify(self, session, specs):
        """net(x, verify=True) populates GraphNode.verification per node."""
        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        net(x, verify=True)
        assert len(net._graph) == 3
        for rec in net._graph:
            v = rec.verification
            assert v != {}
            assert v["passed"] is True
            assert v["max_diff"] == 0.0
            # Each node's declared output tensor is what got verified.
            assert rec.output_name in v["tensors"]

    def test_verification_empty_without_verify(self, session, specs):
        """No verify → no per-node verification recorded."""
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))
        for rec in net._graph:
            assert rec.verification == {}

    def test_verify_report_structure(self, session, specs):
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        report = net.verify_report()
        assert report["all_nodes_passed"] is True
        assert report["passed"] is True
        assert report["e2e"] is None  # no reference supplied
        node_names = [row["node"] for row in report["nodes"]]
        assert node_names == ["scale", "off1", "off2"]
        assert all(row["verified"] and row["passed"] for row in report["nodes"])


class TestEndToEndVerification:

    def test_e2e_passes_vs_correct_reference(self, session, specs):
        """Correct torch reference → E2E verify passes, no raise."""
        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        net(x, verify=True, reference=_reference)  # must not raise
        report = net.verify_report()
        assert report["e2e"]["passed"] is True
        assert report["passed"] is True
        assert len(report["e2e"]["outputs"]) == 2

    def test_e2e_raises_vs_wrong_reference(self, session, specs):
        """Wrong reference → VerificationError with clear message."""
        from vten.errors import VerificationError

        def wrong_reference(x):
            return (
                torch.zeros(N, dtype=torch.int8),
                torch.zeros(N, dtype=torch.int8),
            )

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        with pytest.raises(VerificationError, match="E2E"):
            net(x, verify=True, reference=wrong_reference)

    def test_e2e_accepts_tensor_reference(self, session, specs):
        """A precomputed reference tensor (not callable) also works."""
        class OneOut(InferenceModel):
            def __init__(self, session, specs):
                super().__init__(session)
                self._specs = specs

            def build(self):
                self.scale = self.stage(
                    ScaleKernel, scale_factor=SCALE_FACTOR, N=N,
                    input_name="data_in", output_name="data_out",
                    name="scale", _spec=self._specs["scale"],
                )

            def forward(self, x):
                return self.scale(x)

        net = OneOut(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        golden = (x.to(torch.int16) * SCALE_FACTOR).clamp(-128, 127).to(torch.int8)
        net(x, verify=True, reference=golden)  # must not raise
        assert net.verify_report()["e2e"]["passed"] is True

    def test_e2e_output_count_mismatch_raises(self, session, specs):
        """Reference producing wrong number of outputs → error."""
        from vten.errors import VerificationError

        def one_output_ref(x):
            return torch.zeros(N, dtype=torch.int8)  # model returns 2

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        with pytest.raises(VerificationError):
            net(x, verify=True, reference=one_output_ref)


# ── M1.2 S3: quant-aware verification — lsb_tol plumbing ──


class TestLsbTolPlumbing:
    """net(x, verify=True, lsb_tol=...) reaches every node + the E2E check."""

    def test_scalar_lsb_tol_reaches_execute_batch(
        self, session, specs, monkeypatch,
    ):
        """Model-level scalar lsb_tol flows node → session.run → execute_batch."""
        import vten.execution as vex

        seen: list[object] = []
        orig = vex.execute_batch

        def spy(**kwargs):
            seen.append(kwargs.get("lsb_tolerance"))
            return orig(**kwargs)

        monkeypatch.setattr(vex, "execute_batch", spy)
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=1)
        assert seen == [1, 1, 1]  # scale, off1, off2

    def test_default_lsb_tol_is_zero(self, session, specs, monkeypatch):
        """Without opt-in, execute_batch sees lsb_tolerance=0 (unchanged)."""
        import vten.execution as vex

        seen: list[object] = []
        orig = vex.execute_batch

        def spy(**kwargs):
            seen.append(kwargs.get("lsb_tolerance"))
            return orig(**kwargs)

        monkeypatch.setattr(vex, "execute_batch", spy)
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        assert seen == [0, 0, 0]

    def test_dict_lsb_tol_keyed_by_node_name(self, session, specs, monkeypatch):
        """Dict form resolves per node; unlisted nodes stay bit-exact."""
        import vten.execution as vex

        seen: list[object] = []
        orig = vex.execute_batch

        def spy(**kwargs):
            seen.append(kwargs.get("lsb_tolerance"))
            return orig(**kwargs)

        monkeypatch.setattr(vex, "execute_batch", spy)
        net = FanOutNet(session, specs)
        net(
            torch.arange(-16, 16, dtype=torch.int8),
            verify=True, lsb_tol={"scale": 2, "off2": 1},
        )
        assert seen == [2, 0, 1]  # scale, off1, off2

    def test_e2e_one_lsb_off_fails_without_tol(self, session, specs):
        """Default unchanged: 1-LSB E2E deviation still raises."""
        from vten.errors import VerificationError

        def off_by_one(x_cpu):
            a, b = _reference(x_cpu)
            return a + 1, b

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        with pytest.raises(VerificationError, match="E2E"):
            net(x, verify=True, reference=off_by_one)

    def test_e2e_one_lsb_off_passes_with_tol(self, session, specs):
        """Scalar lsb_tol=1 covers the E2E check; max_lsb_err recorded."""

        def off_by_one(x_cpu):
            a, b = _reference(x_cpu)
            return a + 1, b

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        net(x, verify=True, reference=off_by_one, lsb_tol=1)  # must not raise
        report = net.verify_report()
        assert report["e2e"]["passed"] is True
        outs = {o["name"]: o for o in report["e2e"]["outputs"]}
        assert outs["output[0]"]["max_lsb_err"] == 1
        assert outs["output[1]"]["max_lsb_err"] == 0

    def test_e2e_two_lsb_off_fails_with_tol_one(self, session, specs):
        from vten.errors import VerificationError

        def off_by_two(x_cpu):
            a, b = _reference(x_cpu)
            return a + 2, b

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        with pytest.raises(VerificationError) as exc_info:
            net(x, verify=True, reference=off_by_two, lsb_tol=1)
        assert exc_info.value.max_lsb_err == 2


# ── M1.2 S5: per-node quant-error metrics in verify_report ──


def _offset_session(offsets):
    """InferenceSession over a cpu backend whose k-th kernel run deviates
    from golden by ``offsets[k]`` LSBs (the S3 off-by-one backend idiom,
    generalized to a per-run schedule so nodes get distinct errors)."""
    from vten.backend.cpu import CpuBackend

    class _ScheduledOffsetBackend(CpuBackend):
        """CPU backend: run k's outputs are golden + offsets[k]."""

        def __init__(self, offsets):
            super().__init__()
            self._offsets = list(offsets)

        def execute(self, compiled):
            result = super().execute(compiled)
            off = self._offsets.pop(0) if self._offsets else 0
            if off and result._forward_tensors:
                result._forward_tensors = {
                    k: v + off for k, v in result._forward_tensors.items()
                }
            return result

    return InferenceSession(
        _ScheduledOffsetBackend(offsets),
        project_dir=str(_SCALE_ADD_DIR), log_level="WARNING",
    )


class TestQuantErrorReport:
    """verify_report(): quant-error columns + worst-node aggregate (S5)."""

    def test_exact_rows_on_cpu(self, session, specs):
        """cpu golden == hw → every verified node is max_lsb_err=0 'exact'."""
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        report = net.verify_report()
        for row in report["nodes"]:
            assert row["max_lsb_err"] == 0
            assert row["lsb_tol"] == 0
            assert row["quant_status"] == "exact"

    def test_within_tol_columns(self, specs):
        """A 1-LSB-off node under lsb_tol=1 shows err/tol/'within-tol'."""
        net = FanOutNet(_offset_session([0, 1, 0]), specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=1)
        rows = {r["node"]: r for r in net.verify_report()["nodes"]}
        assert rows["off1"]["max_lsb_err"] == 1
        assert rows["off1"]["lsb_tol"] == 1
        assert rows["off1"]["quant_status"] == "within-tol"
        assert rows["scale"]["quant_status"] == "exact"
        assert rows["off2"]["max_lsb_err"] == 0

    def test_dict_lsb_tol_column_per_node(self, specs):
        """Dict-form tolerance shows up per node; unlisted nodes stay 0."""
        net = FanOutNet(_offset_session([0, 1, 0]), specs)
        net(
            torch.arange(-16, 16, dtype=torch.int8),
            verify=True, lsb_tol={"off1": 1},
        )
        rows = {r["node"]: r for r in net.verify_report()["nodes"]}
        assert rows["off1"]["lsb_tol"] == 1
        assert rows["scale"]["lsb_tol"] == 0
        assert rows["off2"]["lsb_tol"] == 0

    def test_worst_node_aggregate(self, specs):
        """Worst node by max_lsb_err — the quant analogue of bottleneck_node."""
        net = FanOutNet(_offset_session([1, 2, 0]), specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=2)
        report = net.verify_report()
        assert report["worst_quant_node"] == "off1"
        worst = report.worst_quant_node()
        assert worst["node"] == "off1"
        assert worst["max_lsb_err"] == 2
        assert (
            "worst quant error: off1 (max_lsb_err=2, lsb_tol=2)"
            in str(report)
        )

    def test_table_rendering_columns(self, specs):
        net = FanOutNet(_offset_session([0, 1, 0]), specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=1)
        text = str(net.verify_report())
        assert "per-node verification" in text
        for col in ("lsb_err", "lsb_tol", "status"):
            assert col in text
        assert "within-tol" in text and "exact" in text
        assert "scale" in text and "off1" in text and "off2" in text

    def test_e2e_rows(self, specs):
        """E2E checks carry lsb_tol/quantized and render as table rows."""
        net = FanOutNet(_offset_session([0, 1, 0]), specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        net(x, verify=True, reference=_reference, lsb_tol=1)
        report = net.verify_report()
        outs = {o["name"]: o for o in report["e2e"]["outputs"]}
        assert outs["output[0]"]["max_lsb_err"] == 1
        assert outs["output[0]"]["lsb_tol"] == 1
        assert outs["output[0]"]["quantized"] is True
        assert outs["output[1]"]["max_lsb_err"] == 0
        text = str(report)
        assert "E2E output[0]" in text and "E2E output[1]" in text

    def test_e2e_fail_renders_fail_status(self, session, specs):
        from vten.errors import VerificationError

        def wrong_reference(x):
            return (
                torch.zeros(N, dtype=torch.int8),
                torch.zeros(N, dtype=torch.int8),
            )

        net = FanOutNet(session, specs)
        x = torch.arange(-16, 16, dtype=torch.int8)
        with pytest.raises(VerificationError):
            net(x, verify=True, reference=wrong_reference)
        report = net.verify_report()
        assert report["e2e"]["passed"] is False
        assert "FAIL" in str(report)

    def test_unverified_run_degrades(self, session, specs):
        """Plain run: quant columns are None / '—', no worst-node line."""
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))  # no verify
        report = net.verify_report()
        for row in report["nodes"]:
            assert row["max_lsb_err"] is None
            assert row["lsb_tol"] is None
            assert row["quant_status"] is None
        assert report["worst_quant_node"] is None
        text = str(report)
        assert "—" in text
        assert "worst quant error" not in text
        assert net.quant_error_summary() == []

    def test_float_output_marked_unquantized(self, session, specs, monkeypatch):
        """A float-dtype output carries no LSB metrics (columns degrade)."""
        from vten.kernel.tensor import Tensor as VTensor
        from vten.runtime.reporting import VerificationResult

        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        rec = net._graph[0]
        monkeypatch.setattr(
            net._session, "last_verification_results",
            lambda: [VerificationResult(tensor_name=rec.output_name, passed=True)],
        )
        float_out = VTensor(shape=(N,), dtype=torch.float32, interface="m_axis")
        net._capture_node_verification(rec, float_out)
        assert rec.verification["quantized"] is False
        row = next(
            r for r in net.verify_report()["nodes"] if r["node"] == "scale"
        )
        assert row["verified"] is True
        assert row["max_lsb_err"] is None
        assert row["lsb_tol"] is None
        assert row["quant_status"] is None
        assert all(s["name"] != "scale" for s in net.quant_error_summary())

    def test_report_is_json_round_trippable(self, session, specs):
        """Still a plain dict for serializing consumers (Slice-C contract)."""
        import json

        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        report = net.verify_report()
        assert isinstance(report, dict)
        assert json.loads(json.dumps(report)) == report


class TestQuantErrorSummary:
    """net.quant_error_summary() — structured per-node metrics (M2 feed)."""

    def test_summary_contents(self, specs):
        net = FanOutNet(_offset_session([1, 2, 0]), specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=2)
        summary = net.quant_error_summary()
        assert [s["name"] for s in summary] == ["scale", "off1", "off2"]
        assert all(
            set(s) == {"name", "max_lsb_err", "lsb_tol", "passed"}
            for s in summary
        )
        by_name = {s["name"]: s for s in summary}
        assert by_name["scale"] == {
            "name": "scale", "max_lsb_err": 1, "lsb_tol": 2, "passed": True,
        }
        assert by_name["off1"]["max_lsb_err"] == 2
        assert by_name["off2"]["max_lsb_err"] == 0

    def test_summary_picks_worst_node(self, specs):
        net = FanOutNet(_offset_session([1, 2, 0]), specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True, lsb_tol=2)
        worst = max(
            net.quant_error_summary(), key=lambda s: s["max_lsb_err"],
        )
        assert worst["name"] == "off1"
        assert worst["name"] == net.verify_report()["worst_quant_node"]


# ── Slice D: per-node perf rollup (synthetic CmdStats fixtures) ──


@dataclass
class _StubCmdStats:
    """Minimal CmdStats stub (mirrors tests/test_reporting.py)."""

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


def _push_cmd_stats(active, window_start, window_end, latency, beats, stall=0):
    """A single data-moving command's synthetic stats."""
    return _StubCmdStats(
        cmd_id=0, status=3, issue_cycle=window_start,
        commit_cycle=window_start + latency,
        first_active_cycle=window_start, last_active_cycle=window_end,
        active_cycles=active, total_beats=beats, stall_cycles=stall,
    )


class TestPerNodePerf:
    """perf_report() rollup — driven by synthetic CmdStats (cpu has none)."""

    def _model_with_stats(self, session, specs, per_node_stats):
        """Build a FanOutNet and stamp synthetic PerfSummary onto each node.

        We can't get real cycles from cpu, so we mirror test_reporting.py and
        inject stats directly, then exercise the real perf_report() rollup.
        """
        from vten.runtime.reporting import build_perf_summary

        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))
        for rec, cmds in zip(net._graph, per_node_stats):
            # Build enriched command dicts the way the model does, then roll up.
            enriched = [
                {
                    "op": "PUSH", "interface": "s_axis",
                    "protocol": "axi4_stream",
                    "total_beats": c.total_beats,
                    "active_cycles": c.active_cycles,
                    "stall_cycles": c.stall_cycles,
                    "latency_cycles": c.latency_cycles,
                    "first_active_cycle": c.first_active_cycle,
                    "last_active_cycle": c.last_active_cycle,
                    "size": 1024,
                }
                for c in cmds
            ]
            rec.stats = build_perf_summary(enriched)
        return net

    def test_perf_report_builds_per_node_table(self, session, specs):
        net = self._model_with_stats(
            session, specs,
            per_node_stats=[
                [_push_cmd_stats(40, 0, 49, 100, 10)],   # scale
                [_push_cmd_stats(20, 0, 29, 50, 8)],     # off1
                [_push_cmd_stats(30, 0, 39, 60, 12)],    # off2
            ],
        )
        report = net.perf_report()
        assert report.has_data
        d = report.to_dict()
        names = [row["node"] for row in d["nodes"]]
        assert names == ["scale", "off1", "off2"]
        scale_row = d["nodes"][0]
        assert scale_row["kernel"] == "ScaleKernel"
        assert scale_row["active_cycles"] == 40
        assert scale_row["active_window"] == 50  # 49 - 0 + 1
        assert scale_row["total_beats"] == 10
        assert scale_row["bytes_moved"] == 1024
        # Totals roll up across nodes.
        assert d["totals"]["active_cycles"] == 90
        assert d["totals"]["total_beats"] == 30

    def test_bottleneck_node_is_highest_active_cycles(self, session, specs):
        net = self._model_with_stats(
            session, specs,
            per_node_stats=[
                [_push_cmd_stats(40, 0, 49, 100, 10)],   # scale — most cycles
                [_push_cmd_stats(20, 0, 29, 50, 8)],     # off1
                [_push_cmd_stats(30, 0, 39, 60, 12)],    # off2
            ],
        )
        report = net.perf_report()
        assert report.bottleneck_node().node == "scale"
        assert report.to_dict()["bottleneck_node"] == "scale"

    def test_perf_report_terminal_string(self, session, specs):
        net = self._model_with_stats(
            session, specs,
            per_node_stats=[
                [_push_cmd_stats(40, 0, 49, 100, 10)],
                [_push_cmd_stats(20, 0, 29, 50, 8)],
                [_push_cmd_stats(30, 0, 39, 60, 12)],
            ],
        )
        text = str(net.perf_report())
        assert "per-node performance" in text
        assert "scale" in text and "off1" in text and "off2" in text
        assert "bottleneck node: scale" in text
        assert "TOTAL" in text

    def test_perf_report_graceful_without_stats(self, session, specs):
        """cpu backend: no CmdStats → empty report, clear note, no crash."""
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8))
        report = net.perf_report()
        assert not report.has_data
        assert report.bottleneck_node() is None
        assert "no per-node performance data" in str(report)
        d = report.to_dict()
        assert d["nodes"] == []
        assert d["bottleneck_node"] is None

    def test_node_stats_is_none_on_cpu(self, session, specs):
        """Each GraphNode.stats is None on the cpu backend (no CmdStats)."""
        net = FanOutNet(session, specs)
        net(torch.arange(-16, 16, dtype=torch.int8), verify=True)
        for rec in net._graph:
            assert rec.stats is None
