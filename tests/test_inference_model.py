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

    def test_graphnode_has_empty_hook_slots(self, session, specs):
        """verification/stats slots exist and start empty for later agents."""
        net = FanOutNet(session, specs)
        net(torch.zeros(N, dtype=torch.int8))
        for rec in net._graph:
            assert rec.verification == {}
            assert rec.stats == {}

    def test_forward_not_implemented_by_base(self, session):
        net = InferenceModel(session)
        with pytest.raises(NotImplementedError):
            net(torch.zeros(N, dtype=torch.int8))
