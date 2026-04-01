"""Tests for vten.kernel.composite — CompositeKernel, TensorRef, Connection, auto-expose.

Spec reference: 01_kernel_and_dsl.md §4, 00_data_models.md §4, 10_kernel_v2_design.md §5

v2 API: Sub-kernels declared as Kernel(), connections via >> operator,
unconnected tensors auto-exposed.

NPU 3D = CompositeKernel with 6 sub-kernels:
  fmapIO        — ctrl, ddr (exposed), ifm_out/ofm_in (connected)
  weight_loader — ctrl, hbm (exposed), wgt_out (connected)
  mac_atu       — ctrl, ifm_in/wgt_in/psum_out (connected)
  psum_buffer   — ctrl, psum_in/out (connected)
  bias_loader   — ctrl, ddr (exposed), bias_out (connected)
  act_quant     — ctrl, psum_in/bias_in/ofm_out (connected)

6 Internal connections:
  fmapIO.ifm_out → mac_atu.ifm_in
  weight_loader.wgt_out → mac_atu.wgt_in
  mac_atu.psum_out → psum_buffer.psum_in
  psum_buffer.out → act_quant.psum_in
  bias_loader.bias_out → act_quant.bias_in
  act_quant.ofm_out → fmapIO.ofm_in
"""

from __future__ import annotations

import pytest
import torch

from vten.kernel.tensor import Tensor
from vten.kernel.base import Kernel, register
from vten.kernel.composite import (
    CompositeKernel,
    TensorRef,
    Connection,
)


# ── NPU 3D sub-kernels ──────────────────────────────────────────────


class FmapIOKernel(Kernel):
    spec = "fmapIO.yaml"
    ifm = Tensor(shape=("${IN_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
                 dtype=torch.int8, interface="ddr")
    ofm = Tensor(shape=("${OUT_CH}", "${OUT_DEPTH}", "${OUT_HEIGHT}", "${OUT_WIDTH}"),
                 dtype=torch.int8, interface="ddr")
    concat = Tensor(shape=("${CONCAT_CH}", "${IN_DEPTH}", "${IN_HEIGHT}", "${IN_WIDTH}"),
                    dtype=torch.int8, interface="ddr")
    ifm_out = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ifm_out")
    ofm_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="ofm_in")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.ifm.fill_random(generator=torch.Generator().manual_seed(seed or 0))
    def forward(self, **inputs):
        return {"ifm_out": inputs.get("ifm", torch.tensor(0))}


class WeightLoaderKernel(Kernel):
    spec = "weight_loader.yaml"
    weight = Tensor(shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
                    dtype=torch.int8, interface="hbm")
    wgt_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="wgt_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.weight.fill_random(generator=torch.Generator().manual_seed(seed or 0))
    def forward(self, **inputs):
        return {"wgt_out": inputs.get("weight", torch.tensor(0))}


class MacAtuKernel(Kernel):
    spec = "mac_atu.yaml"
    ifm_in = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ifm_in")
    wgt_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="wgt_in")
    psum_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self, **inputs):
        return {"psum_out": torch.tensor(0)}


class PsumBufferKernel(Kernel):
    spec = "psum_buffer.yaml"
    psum_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_in")
    out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self, **inputs):
        return {"out": torch.tensor(0)}


class BiasLoaderKernel(Kernel):
    spec = "bias_loader.yaml"
    bias = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
    bias_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="bias_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.bias.fill_random(generator=torch.Generator().manual_seed(seed or 0))
    def forward(self, **inputs):
        return {"bias_out": inputs.get("bias", torch.tensor(0))}


class ActQuantKernel(Kernel):
    spec = "act_quant.yaml"
    psum_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_in")
    bias_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="bias_in")
    ofm_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="ofm_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self, **inputs):
        return {"ofm_out": torch.tensor(0)}


# ═══════════════════════════════════════════════════════════════════
# §1  TensorRef — created by Kernel.__getattr__ in composite body
# ═══════════════════════════════════════════════════════════════════


class TestTensorRef:

    def test_rshift_creates_connection(self):
        """TensorRef >> TensorRef → Connection."""
        ref_a = TensorRef("src", "data_out", FmapIOKernel)
        ref_b = TensorRef("dst", "data_in", MacAtuKernel)
        conn = ref_a >> ref_b
        assert isinstance(conn, Connection)
        assert conn.source_sub == "src"
        assert conn.dest_sub == "dst"

    def test_rshift_non_tensorref_raises(self):
        ref = TensorRef("src", "data", FmapIOKernel)
        with pytest.raises(TypeError, match="TensorRef"):
            ref >> "not_a_ref"

    def test_repr(self):
        ref = TensorRef("fmapIO", "ifm", FmapIOKernel)
        assert "fmapIO.ifm" in repr(ref)


# ═══════════════════════════════════════════════════════════════════
# §2  Connection — properties
# ═══════════════════════════════════════════════════════════════════


class TestConnection:

    def test_source_dest_subs(self):
        ref_s = TensorRef("fmapIO", "ifm_out", FmapIOKernel)
        ref_d = TensorRef("mac_atu", "ifm_in", MacAtuKernel)
        conn = ref_s >> ref_d
        assert conn.source_sub == "fmapIO"
        assert conn.dest_sub == "mac_atu"
        assert conn.source_name == "ifm_out"
        assert conn.dest_name == "ifm_in"

    def test_source_interface(self):
        ref_s = TensorRef("fmapIO", "ifm_out", FmapIOKernel)
        ref_d = TensorRef("mac_atu", "ifm_in", MacAtuKernel)
        conn = ref_s >> ref_d
        assert conn.source_interface == "ifm_out"

    def test_dest_interface(self):
        ref_s = TensorRef("fmapIO", "ifm_out", FmapIOKernel)
        ref_d = TensorRef("mac_atu", "ifm_in", MacAtuKernel)
        conn = ref_s >> ref_d
        assert conn.dest_interface == "ifm_in"


# ═══════════════════════════════════════════════════════════════════
# §3  Connection via >> operator in composite body
# ═══════════════════════════════════════════════════════════════════


class TestConnectionViaOperator:

    def test_fmapio_to_mac(self):
        """fmapIO.ifm_out >> mac_atu.ifm_in via __getattr__."""
        class TestComposite(CompositeKernel):
            fmapIO = FmapIOKernel()
            mac_atu = MacAtuKernel()
            connections = [fmapIO.ifm_out >> mac_atu.ifm_in]

        conn = TestComposite.connections[0]
        assert conn.source_sub == "fmapIO"
        assert conn.source_name == "ifm_out"
        assert conn.dest_sub == "mac_atu"
        assert conn.dest_name == "ifm_in"

    def test_weight_to_mac(self):
        class TestComposite(CompositeKernel):
            wgt = WeightLoaderKernel()
            mac = MacAtuKernel()
            connections = [wgt.wgt_out >> mac.wgt_in]

        conn = TestComposite.connections[0]
        assert conn.source_sub == "wgt"
        assert conn.dest_sub == "mac"


# ═══════════════════════════════════════════════════════════════════
# §4  NPU3DKernel CompositeKernel — 6-IP structure
# ═══════════════════════════════════════════════════════════════════


class NPU3DKernel(CompositeKernel):
    """NPU 3D — 6 sub-kernels, 6 internal connections."""
    spec = "npu_3d.yaml"

    fmapIO = FmapIOKernel()
    weight_loader = WeightLoaderKernel()
    mac_atu = MacAtuKernel()
    psum_buffer = PsumBufferKernel()
    bias_loader = BiasLoaderKernel()
    act_quant = ActQuantKernel()

    connections = [
        fmapIO.ifm_out >> mac_atu.ifm_in,
        weight_loader.wgt_out >> mac_atu.wgt_in,
        mac_atu.psum_out >> psum_buffer.psum_in,
        psum_buffer.out >> act_quant.psum_in,
        bias_loader.bias_out >> act_quant.bias_in,
        act_quant.ofm_out >> fmapIO.ofm_in,
    ]


class TestNPU3DCompositeDeclaration:

    def test_six_sub_kernels(self):
        """6개 sub-kernel refs가 모두 존재."""
        assert set(NPU3DKernel._sub_kernel_refs.keys()) == {
            "fmapIO", "weight_loader", "mac_atu",
            "psum_buffer", "bias_loader", "act_quant",
        }

    def test_sub_kernel_refs_are_classes(self):
        for name, cls in NPU3DKernel._sub_kernel_refs.items():
            assert isinstance(cls, type), f"{name} should be a class"

    def test_six_connections(self):
        """6개 internal connections."""
        assert len(NPU3DKernel._connections) == 6
        for conn in NPU3DKernel._connections:
            assert isinstance(conn, Connection)

    def test_connection_chain(self):
        """Pipeline: fmapIO→mac→psum→act→fmapIO 순환 구조."""
        conns = NPU3DKernel._connections
        chain = [(c.source_sub, c.dest_sub) for c in conns]
        assert ("fmapIO", "mac_atu") in chain
        assert ("weight_loader", "mac_atu") in chain
        assert ("mac_atu", "psum_buffer") in chain
        assert ("psum_buffer", "act_quant") in chain
        assert ("bias_loader", "act_quant") in chain
        assert ("act_quant", "fmapIO") in chain  # feedback

    def test_connected_tensors(self):
        """Connected tensor set correctly computed."""
        ct = NPU3DKernel._connected_tensors
        assert ("fmapIO", "ifm_out") in ct
        assert ("mac_atu", "ifm_in") in ct
        assert ("act_quant", "ofm_out") in ct
        assert ("fmapIO", "ofm_in") in ct

    def test_auto_exposed_tensors(self):
        """Tensors NOT in connections are auto-exposed."""
        ae = NPU3DKernel._auto_exposed
        # ifm, ofm, concat should be auto-exposed (not in connections)
        assert ("fmapIO", "ifm") in ae
        assert ("fmapIO", "ofm") in ae
        assert ("fmapIO", "concat") in ae
        # weight should be auto-exposed
        assert ("weight_loader", "weight") in ae
        # bias should be auto-exposed
        assert ("bias_loader", "bias") in ae
        # Connected tensors should NOT be auto-exposed
        assert ("fmapIO", "ifm_out") not in ae
        assert ("mac_atu", "ifm_in") not in ae


# ═══════════════════════════════════════════════════════════════════
# §5  CompositeKernel is-a Kernel
# ═══════════════════════════════════════════════════════════════════


class TestCompositeIsKernel:

    def test_isinstance(self):
        assert isinstance(NPU3DKernel(), Kernel)

    def test_has_spec(self):
        assert NPU3DKernel.spec == "npu_3d.yaml"


# ═══════════════════════════════════════════════════════════════════
# §6  Connection validation — protocol, dtype, coverage, duplicates
# ═══════════════════════════════════════════════════════════════════


class TestConnectionValidation:
    """Connection validation through RuntimeEngine._validate_*()."""

    def _make_simple_kernels(self):
        class SrcKernel(Kernel):
            spec = "src.yaml"
            data_out = Tensor(shape=(8,), dtype=torch.int8, interface="output_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {"data_out": torch.zeros(8)}

        class DstKernel(Kernel):
            spec = "dst.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {}

        return SrcKernel, DstKernel

    def _make_dtype_mismatch_kernels(self):
        class SrcF32(Kernel):
            spec = "src.yaml"
            data_out = Tensor(shape=(8,), dtype=torch.float32, interface="output_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {"data_out": torch.zeros(8)}

        class DstI8(Kernel):
            spec = "dst.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {}

        return SrcF32, DstI8

    def test_dtype_mismatch_allowed_for_internal_wires(self):
        """Internal wire connections allow dtype mismatch (physical bytes)."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import KernelInstance
        from vten.spec.models import KernelSpec

        SrcF32, DstI8 = self._make_dtype_mismatch_kernels()

        class DtypeMixComposite(CompositeKernel):
            spec = "mix.yaml"
            src = SrcF32()
            dst = DstI8()
            connections = [src.data_out >> dst.data_in]

        conn = DtypeMixComposite.connections[0]

        # Build minimal sub-kernel instances for validation
        sub_kernels = {}
        for name, cls in [("src", SrcF32), ("dst", DstI8)]:
            ki = KernelInstance(
                name=name,
                spec=KernelSpec(kernel_name=cls.__name__, rtl_top=cls.__name__),
                kernel_class=cls,
            )
            ki.kernel_class_instance = cls()
            from vten.runtime.resolver import ParameterResolver
            import copy
            ki._resolver = ParameterResolver({}, {}, {})
            for t in ki.kernel_class_instance.tensors():
                inst_t = copy.copy(t)
                setattr(ki.kernel_class_instance, t.name, inst_t)
                inst_t._resolve_shape(ki._resolver)
            sub_kernels[name] = ki

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        # Internal wire connections skip dtype check — physical bytes on wire
        engine._validate_connection_dtypes([conn], sub_kernels)

    def test_duplicate_connection_source_raises(self):
        """Same source interface in two connections should raise."""
        from vten.runtime.engine import RuntimeEngine

        SrcKernel, DstKernel = self._make_simple_kernels()

        class DstKernel2(Kernel):
            spec = "dst2.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream2")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {}

        class DupSrcComposite(CompositeKernel):
            spec = "dup.yaml"
            src = SrcKernel()
            dst1 = DstKernel()
            dst2 = DstKernel2()
            connections = [
                src.data_out >> dst1.data_in,
                src.data_out >> dst2.data_in,
            ]

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        with pytest.raises(Exception, match="Duplicate connection source"):
            engine._validate_no_duplicate_connections(
                DupSrcComposite.connections, {},
            )


# ═══════════════════════════════════════════════════════════════════
# §7  CompositeKernel.compute_derived_params auto-chain
# ═══════════════════════════════════════════════════════════════════


class TestCompositeComputeDerivedParams:
    """Auto-chain: composite calls each sub-kernel's compute_derived_params."""

    def test_auto_chain_merges_sub_kernel_results(self):
        """Composite compute_derived_params merges all sub-kernel results."""

        class SubA(Kernel):
            spec = "a.yaml"
            data = Tensor(shape=(8,), dtype=torch.int8, interface="s")

            def compute_derived_params(self):
                x = getattr(self, "x", 0)
                return {"a_derived": x * 2}

        class SubB(Kernel):
            spec = "b.yaml"
            data = Tensor(shape=(8,), dtype=torch.int8, interface="s")

            def compute_derived_params(self):
                x = getattr(self, "x", 0)
                return {"b_derived": x + 10}

        class MyComposite(CompositeKernel):
            spec = "comp.yaml"
            a = SubA()
            b = SubB()
            connections = []

        inst = MyComposite()
        inst.x = 5
        result = inst.compute_derived_params()
        assert result == {"a_derived": 10, "b_derived": 15}

    def test_auto_chain_empty_sub_kernels(self):
        """Composite with no sub-kernels returns empty dict."""

        class EmptyComposite(CompositeKernel):
            spec = "empty.yaml"
            connections = []

        result = EmptyComposite().compute_derived_params()
        assert result == {}

    def test_auto_chain_sub_kernel_no_override(self):
        """Sub-kernel without compute_derived_params returns empty dict."""

        class PlainKernel(Kernel):
            spec = "plain.yaml"
            data = Tensor(shape=(8,), dtype=torch.int8, interface="s")

        class WithDerived(Kernel):
            spec = "wd.yaml"
            data = Tensor(shape=(8,), dtype=torch.int8, interface="s")

            def compute_derived_params(self):
                return {"foo": 42}

        class MixedComposite(CompositeKernel):
            spec = "mix.yaml"
            plain = PlainKernel()
            derived = WithDerived()
            connections = []

        result = MixedComposite().compute_derived_params()
        assert result == {"foo": 42}
