"""Tests for vten.kernel.composite — CompositeKernel, SubKernelBinding, proxy chain.

Spec reference: 01_kernel_and_dsl.md §4, 00_data_models.md §4
NPU 3D patterns: npu_3d_analysis.md §2 (6-IP Architecture), §11.1 (CompositeKernel)

NPU 3D = CompositeKernel with 6 sub-kernels:
  fmapIO        — ctrl: External, ddr: External, ifm_out/ofm_in: Internal
  weight_loader — ctrl: External, hbm: External, wgt_out: Internal
  mac_atu       — ctrl: External, ifm_in/wgt_in/psum_out: Internal
  psum_buffer   — ctrl: External, psum_in/psum_out: Internal
  bias_loader   — ctrl: External, ddr: External, bias_out: Internal
  act_quant     — ctrl: External, psum_in/bias_in/ofm_out: Internal

6 Internal connections (npu_3d_analysis.md §2 Internal Interfaces):
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
    SubKernelBinding,
    TensorProxy,
    ExposedTensorDef,
    Internal,
    Connect,
)


# ── NPU 3D sub-kernels (from test_kernel.py) ──────────────────────


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
    def forward(self):
        return self.ifm.to_float()


class WeightLoaderKernel(Kernel):
    spec = "weight_loader.yaml"
    weight = Tensor(shape=("${OUT_CH}", "${IN_CH}", 3, 3, 3),
                    dtype=torch.int8, interface="hbm")
    wgt_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="wgt_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.weight.fill_random(generator=torch.Generator().manual_seed(seed or 0))
    def forward(self):
        return self.weight.to_float()


class MacAtuKernel(Kernel):
    spec = "mac_atu.yaml"
    ifm_in = Tensor(shape=("${IN_CH}",), dtype=torch.int8, interface="ifm_in")
    wgt_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="wgt_in")
    psum_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self):
        return torch.tensor(0)


class PsumBufferKernel(Kernel):
    spec = "psum_buffer.yaml"
    psum_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_in")
    out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self):
        return torch.tensor(0)


class BiasLoaderKernel(Kernel):
    spec = "bias_loader.yaml"
    bias = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="ddr")
    bias_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="bias_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        self.bias.fill_random(generator=torch.Generator().manual_seed(seed or 0))
    def forward(self):
        return self.bias.to_float()


class ActQuantKernel(Kernel):
    spec = "act_quant.yaml"
    psum_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="psum_in")
    bias_in = Tensor(shape=("${OUT_CH}",), dtype=torch.int32, interface="bias_in")
    ofm_out = Tensor(shape=("${OUT_CH}",), dtype=torch.int8, interface="ofm_out")
    ctrl = register("ctrl")

    def generate_inputs(self, seed=None):
        pass
    def forward(self):
        return torch.tensor(0)


# ═══════════════════════════════════════════════════════════════════
# §1  Internal marker
# ═══════════════════════════════════════════════════════════════════


class TestInternal:

    def test_default_no_probe(self):
        i = Internal()
        assert i.probe is False

    def test_probe_enabled(self):
        """mac_atu의 ifm_in에 probe 설정 가능."""
        i = Internal(probe=True)
        assert i.probe is True


# ═══════════════════════════════════════════════════════════════════
# §2  SubKernelBinding — NPU 3D sub-kernel bindings
# ═══════════════════════════════════════════════════════════════════


class TestSubKernelBinding:

    def test_fmapio_binding(self):
        """fmapIO: ctrl=External, ddr=External, AXIS=Internal."""
        binding = SubKernelBinding(
            kernel_class=FmapIOKernel,
            interface_map={
                "ctrl": "ctrl_fmapio",
                "ddr": "ddr_fmap",
                "ifm_out": Internal(),
                "ofm_in": Internal(),
            },
        )
        assert binding.kernel_class is FmapIOKernel
        assert binding.interface_map["ctrl"] == "ctrl_fmapio"
        assert isinstance(binding.interface_map["ifm_out"], Internal)

    def test_getattr_returns_tensor_proxy(self):
        """fmapIO.ifm → TensorProxy (IFM exposed)."""
        binding = SubKernelBinding(kernel_class=FmapIOKernel, interface_map={})
        binding._attr_name = "fmapIO"

        proxy = binding.ifm
        assert isinstance(proxy, TensorProxy)
        assert proxy.binding_attr_name == "fmapIO"
        assert proxy.tensor_name == "ifm"
        assert proxy.kernel_class is FmapIOKernel

    def test_getattr_nonexistent_tensor_raises(self):
        binding = SubKernelBinding(kernel_class=FmapIOKernel, interface_map={})
        binding._attr_name = "fmapIO"
        with pytest.raises(AttributeError, match="no tensor"):
            _ = binding.nonexistent

    def test_getattr_private_raises(self):
        binding = SubKernelBinding(kernel_class=FmapIOKernel, interface_map={})
        with pytest.raises(AttributeError):
            _ = binding._private

    def test_known_fields_accessible(self):
        binding = SubKernelBinding(
            kernel_class=FmapIOKernel,
            interface_map={"ctrl": "x"},
            params={"IN_CH": 64},
        )
        assert binding.kernel_class is FmapIOKernel
        assert binding.interface_map == {"ctrl": "x"}
        assert binding.params == {"IN_CH": 64}


# ═══════════════════════════════════════════════════════════════════
# §3  TensorProxy — expose (NPU 3D exposed tensors)
# ═══════════════════════════════════════════════════════════════════


class TestTensorProxy:

    def test_expose_ifm_to_ddr(self):
        """fmapIO.ifm → expose to 'ddr_fmap' top interface."""
        proxy = TensorProxy("fmapIO", "ifm", FmapIOKernel)
        exposed = proxy.expose(interface="ddr_fmap")
        assert isinstance(exposed, ExposedTensorDef)
        assert exposed.origin_sub_kernel == "fmapIO"
        assert exposed.origin_name == "ifm"
        assert exposed.top_interface == "ddr_fmap"

    def test_expose_weight_to_hbm(self):
        """weight_loader.weight → expose to 'hbm_wgt'."""
        proxy = TensorProxy("weight_loader", "weight", WeightLoaderKernel)
        exposed = proxy.expose(interface="hbm_wgt")
        assert exposed.origin_sub_kernel == "weight_loader"
        assert exposed.origin_name == "weight"
        assert exposed.top_interface == "hbm_wgt"

    def test_expose_bias_to_ddr(self):
        """bias_loader.bias → expose to 'ddr_bias'."""
        proxy = TensorProxy("bias_loader", "bias", BiasLoaderKernel)
        exposed = proxy.expose(interface="ddr_bias")
        assert exposed.origin_sub_kernel == "bias_loader"
        assert exposed.top_interface == "ddr_bias"


# ═══════════════════════════════════════════════════════════════════
# §4  Connect — NPU 3D internal AXIS connections
# ═══════════════════════════════════════════════════════════════════


class TestConnect:

    def _bind(self, cls, name):
        b = SubKernelBinding(kernel_class=cls, interface_map={})
        b._attr_name = name
        return b

    def test_fmapio_to_mac(self):
        """fmapIO.ifm_out → mac_atu.ifm_in (AXIS 256b)."""
        fmap_b = self._bind(FmapIOKernel, "fmapIO")
        mac_b = self._bind(MacAtuKernel, "mac_atu")
        conn = Connect(fmap_b.ifm_out, mac_b.ifm_in)
        assert conn.source_sub == "fmapIO"
        assert conn.source_name == "ifm_out"
        assert conn.dest_sub == "mac_atu"
        assert conn.dest_name == "ifm_in"

    def test_weight_to_mac(self):
        """weight_loader.wgt_out → mac_atu.wgt_in."""
        wgt_b = self._bind(WeightLoaderKernel, "weight_loader")
        mac_b = self._bind(MacAtuKernel, "mac_atu")
        conn = Connect(wgt_b.wgt_out, mac_b.wgt_in)
        assert conn.source_sub == "weight_loader"
        assert conn.dest_sub == "mac_atu"

    def test_mac_to_psum(self):
        """mac_atu.psum_out → psum_buffer.psum_in."""
        mac_b = self._bind(MacAtuKernel, "mac_atu")
        psum_b = self._bind(PsumBufferKernel, "psum_buffer")
        conn = Connect(mac_b.psum_out, psum_b.psum_in)
        assert conn.source_sub == "mac_atu"
        assert conn.dest_sub == "psum_buffer"

    def test_act_to_fmapio(self):
        """act_quant.ofm_out → fmapIO.ofm_in (feedback loop)."""
        act_b = self._bind(ActQuantKernel, "act_quant")
        fmap_b = self._bind(FmapIOKernel, "fmapIO")
        conn = Connect(act_b.ofm_out, fmap_b.ofm_in)
        assert conn.source_sub == "act_quant"
        assert conn.dest_sub == "fmapIO"
        assert conn.transform is None

    def test_source_interface_extracted(self):
        """Connect는 source tensor의 interface를 추출."""
        fmap_b = self._bind(FmapIOKernel, "fmapIO")
        mac_b = self._bind(MacAtuKernel, "mac_atu")
        conn = Connect(fmap_b.ifm_out, mac_b.ifm_in)
        assert conn.source_interface == "ifm_out"

    def test_non_proxy_source_raises(self):
        with pytest.raises(TypeError, match="source must be TensorProxy"):
            Connect("not_a_proxy", TensorProxy("x", "y", FmapIOKernel))

    def test_non_proxy_dest_raises(self):
        fmap_b = self._bind(FmapIOKernel, "fmapIO")
        with pytest.raises(TypeError, match="dest must be TensorProxy"):
            Connect(fmap_b.ifm_out, "not_a_proxy")


# ═══════════════════════════════════════════════════════════════════
# §5  NPU3DKernel CompositeKernel — 6-IP structure
# ═══════════════════════════════════════════════════════════════════


class NPU3DKernel(CompositeKernel):
    """NPU 3D — 6 sub-kernels, 6 internal connections.
    Per npu_3d_analysis.md §11.1."""
    spec = "npu_3d.yaml"

    # Sub-kernels with interface mappings
    fmapIO = FmapIOKernel.bind(
        interface_map={
            "ctrl": "ctrl_fmapio",
            "ddr": "ddr_fmap",
            "ifm_out": Internal(),
            "ofm_in": Internal(),
        },
    )
    weight_loader = WeightLoaderKernel.bind(
        interface_map={
            "ctrl": "ctrl_wgt",
            "hbm": "hbm_wgt",
            "wgt_out": Internal(),
        },
    )
    mac_atu = MacAtuKernel.bind(
        interface_map={
            "ctrl": "ctrl_mac",
            "ifm_in": Internal(probe=True),
            "wgt_in": Internal(),
            "psum_out": Internal(),
        },
    )
    psum_buffer = PsumBufferKernel.bind(
        interface_map={
            "ctrl": "ctrl_psum",
            "psum_in": Internal(),
            "out": Internal(),
        },
    )
    bias_loader = BiasLoaderKernel.bind(
        interface_map={
            "ctrl": "ctrl_bias",
            "ddr": "ddr_bias",
            "bias_out": Internal(),
        },
    )
    act_quant = ActQuantKernel.bind(
        interface_map={
            "ctrl": "ctrl_act",
            "psum_in": Internal(),
            "bias_in": Internal(),
            "ofm_out": Internal(),
        },
    )

    # Exposed tensors — host ↔ DDR/HBM
    ifm_data = fmapIO.ifm.expose(interface="ddr_fmap")
    ofm_data = fmapIO.ofm.expose(interface="ddr_fmap")
    weight_data = weight_loader.weight.expose(interface="hbm_wgt")
    bias_data = bias_loader.bias.expose(interface="ddr_bias")

    # Internal AXI-Stream connections (no BFM needed)
    connections = [
        Connect(fmapIO.ifm_out, mac_atu.ifm_in),
        Connect(weight_loader.wgt_out, mac_atu.wgt_in),
        Connect(mac_atu.psum_out, psum_buffer.psum_in),
        Connect(psum_buffer.out, act_quant.psum_in),
        Connect(bias_loader.bias_out, act_quant.bias_in),
        Connect(act_quant.ofm_out, fmapIO.ofm_in),
    ]


class TestNPU3DCompositeDeclaration:

    def test_six_sub_kernels(self):
        """6개 sub-kernel 바인딩이 모두 존재."""
        bindings = NPU3DKernel().bindings()
        names = {name for name, _ in bindings}
        assert names == {
            "fmapIO", "weight_loader", "mac_atu",
            "psum_buffer", "bias_loader", "act_quant",
        }

    def test_all_bindings_are_sub_kernel_binding(self):
        for _, binding in NPU3DKernel().bindings():
            assert isinstance(binding, SubKernelBinding)

    def test_six_connections(self):
        """6개 internal AXIS 연결."""
        assert len(NPU3DKernel.connections) == 6
        for conn in NPU3DKernel.connections:
            assert isinstance(conn, Connect)

    def test_connection_chain(self):
        """Pipeline: fmapIO→mac→psum→act→fmapIO 순환 구조."""
        conns = NPU3DKernel.connections
        chain = [(c.source_sub, c.dest_sub) for c in conns]
        assert ("fmapIO", "mac_atu") in chain
        assert ("weight_loader", "mac_atu") in chain
        assert ("mac_atu", "psum_buffer") in chain
        assert ("psum_buffer", "act_quant") in chain
        assert ("bias_loader", "act_quant") in chain
        assert ("act_quant", "fmapIO") in chain  # feedback

    def test_four_exposed_tensors(self):
        """4개 exposed tensor: ifm, ofm, weight, bias."""
        exposed = NPU3DKernel().exposed_tensor_defs()
        names = {name for name, _ in exposed}
        assert names == {"ifm_data", "ofm_data", "weight_data", "bias_data"}

    def test_exposed_ifm_details(self):
        for name, edef in NPU3DKernel().exposed_tensor_defs():
            if name == "ifm_data":
                assert edef.origin_sub_kernel == "fmapIO"
                assert edef.origin_name == "ifm"
                assert edef.top_interface == "ddr_fmap"

    def test_exposed_weight_details(self):
        for name, edef in NPU3DKernel().exposed_tensor_defs():
            if name == "weight_data":
                assert edef.origin_sub_kernel == "weight_loader"
                assert edef.origin_name == "weight"
                assert edef.top_interface == "hbm_wgt"

    def test_mac_ifm_probe_enabled(self):
        """mac_atu.ifm_in에 probe=True — 내부 데이터 검증용."""
        for name, binding in NPU3DKernel().bindings():
            if name == "mac_atu":
                ifm_mapping = binding.interface_map["ifm_in"]
                assert isinstance(ifm_mapping, Internal)
                assert ifm_mapping.probe is True

    def test_external_interfaces(self):
        """External: 6× ctrl + ddr_fmap + hbm_wgt + ddr_bias = 9 interfaces."""
        externals = set()
        for _, binding in NPU3DKernel().bindings():
            for iface_name, mapping in binding.interface_map.items():
                if isinstance(mapping, str):
                    externals.add(mapping)
        expected = {
            "ctrl_fmapio", "ctrl_wgt", "ctrl_mac",
            "ctrl_psum", "ctrl_bias", "ctrl_act",
            "ddr_fmap", "hbm_wgt", "ddr_bias",
        }
        assert externals == expected


# ═══════════════════════════════════════════════════════════════════
# §6  CompositeKernel is-a Kernel
# ═══════════════════════════════════════════════════════════════════


class TestCompositeIsKernel:

    def test_isinstance(self):
        assert isinstance(NPU3DKernel(), Kernel)

    def test_has_spec(self):
        assert NPU3DKernel.spec == "npu_3d.yaml"


# ═══════════════════════════════════════════════════════════════════
# §7  Connection validation — protocol, dtype, coverage, duplicates
# ═══════════════════════════════════════════════════════════════════


class TestConnectDestInterface:
    """Connect should capture dest_interface at construction time."""

    def test_dest_interface_stored(self):
        """Connect(fmapIO.ifm_out, mac_atu.ifm_in) → dest_interface='ifm_in'."""
        conn = Connect(FmapIOKernel.bind(
            interface_map={"ctrl": "c", "ddr": "d", "ifm_out": Internal(), "ofm_in": Internal()},
        ).ifm_out, MacAtuKernel.bind(
            interface_map={"ctrl": "c", "ifm_in": Internal(), "wgt_in": Internal(), "psum_out": Internal()},
        ).ifm_in)
        assert conn.dest_interface == "ifm_in"

    def test_source_interface_stored(self):
        conn = Connect(FmapIOKernel.bind(
            interface_map={"ctrl": "c", "ddr": "d", "ifm_out": Internal(), "ofm_in": Internal()},
        ).ifm_out, MacAtuKernel.bind(
            interface_map={"ctrl": "c", "ifm_in": Internal(), "wgt_in": Internal(), "psum_out": Internal()},
        ).ifm_in)
        assert conn.source_interface == "ifm_out"


class TestConnectionValidation:
    """Connection validation through RuntimeEngine._validate_flattened()."""

    def _make_simple_kernels(self):
        """Create two simple stream kernels for validation testing."""

        class SrcKernel(Kernel):
            spec = "src.yaml"
            data_out = Tensor(shape=(8,), dtype=torch.int8, interface="output_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self): return torch.zeros(8)

        class DstKernel(Kernel):
            spec = "dst.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self): return torch.zeros(8)

        return SrcKernel, DstKernel

    def _make_dtype_mismatch_kernels(self):
        """Create kernels with mismatched dtypes."""

        class SrcF32(Kernel):
            spec = "src.yaml"
            data_out = Tensor(shape=(8,), dtype=torch.float32, interface="output_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self): return torch.zeros(8)

        class DstI8(Kernel):
            spec = "dst.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self): return torch.zeros(8)

        return SrcF32, DstI8

    def test_dtype_mismatch_raises(self):
        """Connecting float32 → int8 without transform should raise."""
        from vten.errors import ConnectionDtypeMismatchError
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import (
            ExposedTensor, InterfaceMapping, KernelInstance,
        )
        from vten.spec.models import Direction, KernelSpec, MappingType

        SrcF32, DstI8 = self._make_dtype_mismatch_kernels()

        # Build a minimal composite with dtype mismatch
        class BadDtypeComposite(CompositeKernel):
            spec = "bad.yaml"
            src = SrcF32.bind(interface_map={
                "ctrl": "ctrl_src",
                "output_stream": Internal(),
            })
            dst = DstI8.bind(interface_map={
                "ctrl": "ctrl_dst",
                "input_stream": Internal(),
            })
            connections = [Connect(src.data_out, dst.data_in)]

        conn = BadDtypeComposite.connections[0]

        # Build minimal sub-kernel instances for validation
        src_ki = KernelInstance(
            name="src",
            spec=KernelSpec(kernel_name="SrcF32", rtl_top="SrcF32"),
            kernel_class=SrcF32,
        )
        src_ki.kernel_class_instance = SrcF32()
        dst_ki = KernelInstance(
            name="dst",
            spec=KernelSpec(kernel_name="DstI8", rtl_top="DstI8"),
            kernel_class=DstI8,
        )
        dst_ki.kernel_class_instance = DstI8()

        # Resolve tensor shapes (trivial — no params)
        from vten.runtime.resolver import ParameterResolver
        for ki in (src_ki, dst_ki):
            ki._resolver = ParameterResolver({}, {}, {})
            import copy
            for t in ki.kernel_class_instance.tensors():
                inst_t = copy.copy(t)
                setattr(ki.kernel_class_instance, t.name, inst_t)
                inst_t._resolve_shape(ki._resolver)

        sub_kernels = {"src": src_ki, "dst": dst_ki}

        engine = RuntimeEngine(
            kernels={}, ops=[], project_params={},
        )

        with pytest.raises(ConnectionDtypeMismatchError, match="dtype mismatch"):
            engine._validate_connection_dtypes(
                [conn], sub_kernels,
            )

    def test_duplicate_connection_source_raises(self):
        """Same source interface in two connections should raise."""
        from vten.runtime.engine import RuntimeEngine

        SrcKernel, DstKernel = self._make_simple_kernels()

        class DstKernel2(Kernel):
            spec = "dst2.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream2")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self): return torch.zeros(8)

        src_binding = SrcKernel.bind(interface_map={
            "ctrl": "c1", "output_stream": Internal(),
        })
        dst_binding = DstKernel.bind(interface_map={
            "ctrl": "c2", "input_stream": Internal(),
        })
        dst2_binding = DstKernel2.bind(interface_map={
            "ctrl": "c3", "input_stream2": Internal(),
        })

        conn1 = Connect(src_binding.data_out, dst_binding.data_in)
        conn2 = Connect(src_binding.data_out, dst2_binding.data_in)

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        with pytest.raises(Exception, match="Duplicate connection source"):
            engine._validate_no_duplicate_connections(
                [conn1, conn2], {},
            )
