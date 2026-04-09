"""Tests for declarative probe API (probes = [...] on TestScenario).

Covers:
  §1 _apply_declarative_probes — output probe annotation
  §2 _apply_declarative_probes — internal probe request collection
  §3 _resolve_internal_probe_golden — golden chain extraction
  §4 _ensure_probe_mappings — INTERNAL → INTERNAL_PROBE upgrade
  §5 TestScenario.probes field
  §6 extract_probe_bfm_info extra_probes parameter
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from vten.dsl.operations import Operation
from vten.runtime.context import ExecutionContext
from vten.spec.models import MappingType, OpKind


# ═══════════════════════════════════════════════════════════════════
# §1  _apply_declarative_probes — output probe
# ═══════════════════════════════════════════════════════════════════


class TestApplyOutputProbe:
    """Output probes: simple name → probe=True on matching PULL ops."""

    def _make_ctx_with_pull(self, tensor_name: str) -> ExecutionContext:
        ctx = ExecutionContext()
        tensor = MagicMock()
        tensor.name = tensor_name
        op = Operation(kind=OpKind.PULL_TENSOR, tensor=tensor)
        ctx._pending_ops.append(op)
        return ctx

    def test_pull_tensor_marked_as_probe(self):
        ctx = self._make_ctx_with_pull("data_out")
        ctx._register_declarative_probes(["data_out"])
        ctx._apply_declarative_probes()
        assert ctx._pending_ops[0].probe is True

    def test_unrelated_tensor_not_marked(self):
        ctx = self._make_ctx_with_pull("data_out")
        ctx._register_declarative_probes(["other_tensor"])
        ctx._apply_declarative_probes()
        assert ctx._pending_ops[0].probe is False

    def test_recv_tensor_also_marked(self):
        ctx = ExecutionContext()
        tensor = MagicMock()
        tensor.name = "data_out"
        op = Operation(kind=OpKind.PULL_TENSOR, tensor=tensor)
        ctx._pending_ops.append(op)
        ctx._register_declarative_probes(["data_out"])
        ctx._apply_declarative_probes()
        assert ctx._pending_ops[0].probe is True

    def test_push_tensor_not_marked(self):
        """PUSH ops should NOT be marked as probe (only PULL/RECV)."""
        ctx = ExecutionContext()
        tensor = MagicMock()
        tensor.name = "data_in"
        op = Operation(kind=OpKind.PUSH_TENSOR, tensor=tensor)
        ctx._pending_ops.append(op)
        ctx._register_declarative_probes(["data_in"])
        ctx._apply_declarative_probes()
        assert ctx._pending_ops[0].probe is False

    def test_no_probes_is_noop(self):
        ctx = self._make_ctx_with_pull("data_out")
        ctx._apply_declarative_probes()  # no probes registered
        assert ctx._pending_ops[0].probe is False


# ═══════════════════════════════════════════════════════════════════
# §2  _apply_declarative_probes — internal probe requests
# ═══════════════════════════════════════════════════════════════════


class TestApplyInternalProbeRequests:
    """Dotted names → stored as internal probe requests."""

    def test_dotted_name_creates_request(self):
        ctx = ExecutionContext()
        ctx._register_declarative_probes(["scale.data_out"])
        ctx._apply_declarative_probes()
        assert ("scale", "data_out") in ctx._internal_probe_requests

    def test_mixed_probes_separated(self):
        ctx = ExecutionContext()
        tensor = MagicMock()
        tensor.name = "data_out"
        op = Operation(kind=OpKind.PULL_TENSOR, tensor=tensor)
        ctx._pending_ops.append(op)

        ctx._register_declarative_probes(["scale.data_out", "data_out"])
        ctx._apply_declarative_probes()

        assert ("scale", "data_out") in ctx._internal_probe_requests
        assert ctx._pending_ops[0].probe is True

    def test_multiple_internal_probes(self):
        ctx = ExecutionContext()
        ctx._register_declarative_probes(["scale.data_out", "offset.data_in"])
        ctx._apply_declarative_probes()
        assert len(ctx._internal_probe_requests) == 2


# ═══════════════════════════════════════════════════════════════════
# §3  _resolve_internal_probe_golden
# ═══════════════════════════════════════════════════════════════════


class TestResolveInternalProbeGolden:
    """Auto-extract golden from CompositeKernel's _golden_pool."""

    def _make_ctx_with_composite(self, golden_pool):
        ctx = ExecutionContext()

        # Mock kernel class instance with _golden_pool
        # v2: pool keys are (sub_name, tensor_name) tuples
        composite_inst = MagicMock()
        composite_inst._golden_pool = golden_pool

        # Mock KernelInstance
        ki = MagicMock()
        ki.kernel_class_instance = composite_inst
        ctx._kernels["MockComposite"] = ki

        return ctx

    def test_golden_extracted_from_pool(self):
        golden_tensor = torch.arange(32, dtype=torch.int8)
        # v2: pool keys are (sub_name, tensor_name) tuples
        pool = {("scale", "data_out"): golden_tensor}

        ctx = self._make_ctx_with_composite(pool)
        ctx._internal_probe_requests.append(("scale", "data_out"))
        ctx._resolve_internal_probe_golden()

        assert ("scale", "data_out") in ctx._internal_probe_golden
        assert torch.equal(ctx._internal_probe_golden[("scale", "data_out")], golden_tensor)

    def test_missing_sub_kernel_skipped(self):
        ctx = self._make_ctx_with_composite({})
        ctx._internal_probe_requests.append(("nonexistent", "data_out"))
        ctx._resolve_internal_probe_golden()
        assert len(ctx._internal_probe_golden) == 0

    def test_missing_golden_key_skipped(self):
        # v2: pool has no matching key
        ctx = self._make_ctx_with_composite({})
        ctx._internal_probe_requests.append(("scale", "data_out"))
        ctx._resolve_internal_probe_golden()
        assert len(ctx._internal_probe_golden) == 0

    def test_manual_golden_not_overwritten(self):
        """If set_internal_probe_golden was already called, don't overwrite."""
        golden_manual = torch.ones(32, dtype=torch.int8)
        golden_pool = torch.zeros(32, dtype=torch.int8)

        # v2: pool keys are (sub_name, tensor_name) tuples
        ctx = self._make_ctx_with_composite(
            {("scale", "data_out"): golden_pool}
        )
        ctx._internal_probe_golden[("scale", "data_out")] = golden_manual
        ctx._internal_probe_requests.append(("scale", "data_out"))
        ctx._resolve_internal_probe_golden()

        # Manual golden should be preserved
        assert torch.equal(ctx._internal_probe_golden[("scale", "data_out")], golden_manual)

    def test_no_golden_pool_skipped(self):
        """Non-composite kernel (no _golden_pool) → no error."""
        ctx = ExecutionContext()
        ki = MagicMock()
        ki.kernel_class_instance = MagicMock(spec=[])  # no _golden_pool
        ctx._kernels["Simple"] = ki
        ctx._internal_probe_requests.append(("scale", "data_out"))
        ctx._resolve_internal_probe_golden()
        assert len(ctx._internal_probe_golden) == 0


# ═══════════════════════════════════════════════════════════════════
# §4  _ensure_probe_mappings — INTERNAL → INTERNAL_PROBE
# ═══════════════════════════════════════════════════════════════════


class TestEnsureProbeMappings:
    """Engine upgrades INTERNAL mappings to INTERNAL_PROBE for declarative probes."""

    def _make_view_and_engine(self):
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.kernel_view import (
            FlattenedKernelView,
            InterfaceMapping,
        )
        from vten.spec.models import KernelSpec

        spec = KernelSpec(kernel_name="test", rtl_top="test.sv")

        # Mock connection
        conn = MagicMock()
        conn.source_sub = "scale"
        conn.source_name = "data_out"
        conn.source_interface = "output_stream"

        # INTERNAL mapping (not INTERNAL_PROBE)
        mapping = InterfaceMapping(
            sub_kernel="scale",
            sub_interface="output_stream",
            top_interface="",
            mapping_type=MappingType.INTERNAL,
            bank_name=None,
            bank_offset=0,
        )

        view = FlattenedKernelView(
            name="composite",
            top_spec=spec,
            sub_kernels={},
            interface_mappings=[mapping],
            exposed_tensors={},
            probe_points=[],
            connections=[conn],
        )

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        return view, engine, mapping

    def test_internal_upgraded_to_probe(self):
        view, engine, mapping = self._make_view_and_engine()
        golden = {("scale", "data_out"): torch.zeros(32, dtype=torch.int8)}

        engine._ensure_probe_mappings(view, golden)

        assert mapping.mapping_type == MappingType.INTERNAL_PROBE
        assert len(view.probe_points) == 1
        assert view.probe_points[0].connection.source_sub == "scale"

    def test_already_probed_not_duplicated(self):
        """If probe point already exists, don't create another."""
        from vten.runtime.kernel_view import ProbePoint

        view, engine, mapping = self._make_view_and_engine()
        mapping.mapping_type = MappingType.INTERNAL_PROBE

        existing_probe = ProbePoint(
            connection=view.connections[0],
            interface_mapping=mapping,
        )
        view.probe_points.append(existing_probe)

        golden = {("scale", "data_out"): torch.zeros(32, dtype=torch.int8)}
        engine._ensure_probe_mappings(view, golden)

        assert len(view.probe_points) == 1  # not duplicated

    def test_no_matching_connection_noop(self):
        view, engine, mapping = self._make_view_and_engine()
        golden = {("nonexistent", "data_out"): torch.zeros(32, dtype=torch.int8)}

        engine._ensure_probe_mappings(view, golden)

        assert mapping.mapping_type == MappingType.INTERNAL  # unchanged
        assert len(view.probe_points) == 0


# ═══════════════════════════════════════════════════════════════════
# §5  TestScenario.probes field
# ═══════════════════════════════════════════════════════════════════


class TestScenarioProbesField:
    """TestScenario has a `probes` attribute."""

    def test_default_none(self):
        from vten.cli.scenario import TestScenario

        ts = TestScenario()
        assert ts.probes is None

    def test_probes_set(self):
        from vten.cli.scenario import TestScenario

        class MyTest(TestScenario):
            kernel = "scale_add"
            probes = ["scale.data_out", "data_out"]

        ts = MyTest()
        assert ts.probes == ["scale.data_out", "data_out"]


# ═══════════════════════════════════════════════════════════════════
# §6  extract_probe_bfm_info extra_probes
# ═══════════════════════════════════════════════════════════════════


class TestExtractProbeBfmInfoExtraProbes:
    """extra_probes parameter resolves tensor → interface for probing."""

    def test_extra_probes_resolves_interface(self):
        """Verify extra_probes adds (sub_name, interface) to probed set.

        v2: Uses Kernel() instances and >> operator instead of bind()/Internal().
        """
        from vten.kernel.base import Kernel
        from vten.kernel.composite import CompositeKernel, Connection
        from vten.kernel.tensor import Tensor

        class SubA(Kernel):
            data_out = Tensor(
                shape=(32,), dtype="int8", interface="output_stream",
            )

        class SubB(Kernel):
            data_in = Tensor(
                shape=(32,), dtype="int8", interface="input_stream",
            )

        class MyComposite(CompositeKernel):
            a = SubA()
            b = SubB()
            connections = [a.data_out >> b.data_in]

        # v2: connected tensors are internal, unconnected are auto-exposed
        # Verify resolution: "a.data_out" → ("a", "output_stream") via Tensor.interface
        sub_refs = MyComposite._sub_kernel_refs

        # extra_probes: resolve tensor name to interface
        extra_probes = ["a.data_out"]
        probed: set[tuple[str, str]] = set()
        for probe_spec in extra_probes:
            sub_name, tensor_name = probe_spec.rsplit(".", 1)
            sub_cls = sub_refs.get(sub_name)
            assert sub_cls is not None
            tensor = sub_cls._tensor_descriptors.get(tensor_name)
            assert isinstance(tensor, Tensor)
            assert tensor.interface == "output_stream"
            probed.add((sub_name, tensor.interface))

        assert ("a", "output_stream") in probed
