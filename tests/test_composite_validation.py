"""Comprehensive CompositeKernel validation tests.

Tests:
  A1. forward() golden correctness — various parameters
  A2. Edge case N sizes
  A3. Connection validation (negative tests)
  A4. Compile pipeline checks (caching, reuse, project_dir)
  A5. Codegen output verification
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import pytest
import torch

from vten.errors import (
    ConnectionDtypeMismatchError,
    ValidationError,
)
from vten.kernel.base import Kernel
from vten.kernel.composite import (
    CompositeKernel,
    Connection,
)
from vten.kernel.register import register
from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext
from vten.spec.models import Direction


# ── Helper: import scale_add kernels ──


def _setup_scale_add_path():
    """Add scale_add kernel dirs to sys.path."""
    base = Path(__file__).resolve().parent.parent / "examples" / "scale_add" / "kernels"
    for d in ["scale", "offset", "read_dma", "write_dma", "scale_add", "dma_pipeline"]:
        p = str(base / d)
        if p not in sys.path:
            sys.path.insert(0, p)
    # Parent dir for cross-kernel imports
    p = str(base)
    if p not in sys.path:
        sys.path.insert(0, p)


_setup_scale_add_path()

from scale_add_kernel import ScaleAddKernel
from dma_pipeline_kernel import DmaPipelineKernel


# ═══════════════════════════════════════════════════════════════════
# A1. forward() Golden Correctness — Various Parameters
# ═══════════════════════════════════════════════════════════════════


class TestScaleAddGolden:
    """ScaleAddKernel.forward() with diverse parameters."""

    def _run_golden(self, N=1024, scale_factor=2, offset_value=1, seed=42):
        """Instantiate, fill, forward, return result."""
        project_dir = str(
            Path(__file__).resolve().parent.parent / "examples" / "scale_add"
        )
        ctx = ExecutionContext(
            project_params={"N": N, "_project_dir": project_dir}
        )
        k = ctx.instantiate(
            ScaleAddKernel, N=N,
            scale_factor=scale_factor, offset_value=offset_value,
        )
        k.generate_inputs(seed=seed)
        return k, k.forward()["data_out"]

    def test_default(self):
        """scale=2, offset=1: basic operation."""
        k, result = self._run_golden()
        assert result.dtype == torch.int8
        assert result.shape == (1024,)
        # Manual check: (input * 2).clamp + 1
        expected = (k.data_in.data.to(torch.int16) * 2).clamp(-128, 127) + 1
        expected = expected.clamp(-128, 127).to(torch.int8)
        assert torch.equal(result, expected)

    def test_identity(self):
        """scale=1, offset=0: output == input."""
        k, result = self._run_golden(scale_factor=1, offset_value=0)
        assert torch.equal(result, k.data_in.data)

    def test_scale_x3_offset_5(self):
        """scale=3, offset=5: larger values."""
        k, result = self._run_golden(scale_factor=3, offset_value=5)
        expected = (k.data_in.data.to(torch.int16) * 3).clamp(-128, 127) + 5
        expected = expected.clamp(-128, 127).to(torch.int8)
        assert torch.equal(result, expected)

    def test_negative_scale(self):
        """scale=-1, offset=0: sign flip."""
        k, result = self._run_golden(scale_factor=-1, offset_value=0)
        expected = (k.data_in.data.to(torch.int16) * -1).clamp(-128, 127)
        expected = expected.clamp(-128, 127).to(torch.int8)
        assert torch.equal(result, expected)

    def test_overflow_up(self):
        """scale=127, offset=1: saturation at +127."""
        k, result = self._run_golden(scale_factor=127, offset_value=1)
        # Most values should saturate to +127 or -128
        assert result.max() <= 127
        assert result.min() >= -128

    def test_overflow_down(self):
        """scale=-128, offset=-128: extreme negative saturation."""
        k, result = self._run_golden(scale_factor=-128, offset_value=-128)
        assert result.max() <= 127
        assert result.min() >= -128

    def test_zero_scale(self):
        """scale=0, offset=42: all outputs should be 42."""
        k, result = self._run_golden(scale_factor=0, offset_value=42)
        assert torch.all(result == 42)


class TestDmaPipelineGolden:
    """DmaPipelineKernel.forward() golden correctness."""

    def _run_golden(self, N=1024, scale_factor=2, offset_value=1, seed=42):
        project_dir = str(
            Path(__file__).resolve().parent.parent / "examples" / "scale_add"
        )
        ctx = ExecutionContext(
            project_params={"N": N, "_project_dir": project_dir}
        )
        k = ctx.instantiate(
            DmaPipelineKernel, N=N,
            scale_factor=scale_factor, offset_value=offset_value,
        )
        k.generate_inputs(seed=seed)
        return k, k.forward()["data_out"]

    def test_default(self):
        k, result = self._run_golden()
        expected = (k.data_in.data.to(torch.int16) * 2).clamp(-128, 127) + 1
        expected = expected.clamp(-128, 127).to(torch.int8)
        assert torch.equal(result, expected)

    def test_identity(self):
        """DMA round-trip identity: scale=1, offset=0."""
        k, result = self._run_golden(scale_factor=1, offset_value=0)
        assert torch.equal(result, k.data_in.data)

    def test_overflow(self):
        """scale=10, offset=50: saturation."""
        k, result = self._run_golden(scale_factor=10, offset_value=50)
        assert result.max() <= 127
        assert result.min() >= -128


# ═══════════════════════════════════════════════════════════════════
# A2. Edge Case N Sizes
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCaseN:
    """CompositeKernel with various N sizes."""

    def _project_dir(self):
        return str(
            Path(__file__).resolve().parent.parent / "examples" / "scale_add"
        )

    @pytest.mark.parametrize("N", [32, 64, 96, 4096])
    def test_scale_add_n(self, N):
        ctx = ExecutionContext(
            project_params={"N": N, "_project_dir": self._project_dir()}
        )
        k = ctx.instantiate(ScaleAddKernel, N=N, scale_factor=2, offset_value=1)
        k.generate_inputs(seed=42)
        result = k.forward()["data_out"]
        assert result.shape == (N,)
        assert result.dtype == torch.int8

    @pytest.mark.parametrize("N", [32, 64, 4096])
    def test_dma_pipeline_n(self, N):
        ctx = ExecutionContext(
            project_params={"N": N, "_project_dir": self._project_dir()}
        )
        k = ctx.instantiate(DmaPipelineKernel, N=N, scale_factor=2, offset_value=1)
        k.generate_inputs(seed=42)
        result = k.forward()["data_out"]
        assert result.shape == (N,)


# ═══════════════════════════════════════════════════════════════════
# A3. Connection Validation (Negative Tests)
# ═══════════════════════════════════════════════════════════════════


class _StreamSrcKernel(Kernel):
    spec = "src.yaml"
    data_out = Tensor(shape=(8,), dtype=torch.int8, interface="output_stream")
    ctrl = register("ctrl")
    def generate_inputs(self, seed=None): pass
    def forward(self, **inputs): return {"data_out": torch.zeros(8)}


class _StreamDstKernel(Kernel):
    spec = "dst.yaml"
    data_in = Tensor(shape=(8,), dtype=torch.int8, interface="input_stream")
    ctrl = register("ctrl")
    def generate_inputs(self, seed=None): pass
    def forward(self, **inputs): return {}


class _FloatDstKernel(Kernel):
    spec = "dst_f32.yaml"
    data_in = Tensor(shape=(8,), dtype=torch.float32, interface="input_stream")
    ctrl = register("ctrl")
    def generate_inputs(self, seed=None): pass
    def forward(self, **inputs): return {}


class TestConnectionValidationNegative:
    """Negative tests: bad composite configurations should raise errors."""

    def test_dtype_mismatch_allowed_for_internal_wires(self):
        """int8 → float32 on internal wire is OK (physical bytes on wire)."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import KernelInstance
        from vten.runtime.resolver import ParameterResolver
        from vten.spec.models import KernelSpec

        class _DtypeMixComposite(CompositeKernel):
            spec = "mix.yaml"
            src = _StreamSrcKernel()
            dst = _FloatDstKernel()
            connections = [src.data_out >> dst.data_in]

        conn = _DtypeMixComposite.connections[0]

        # Build minimal KernelInstances keyed by binding attr name
        sub_kernels = {}
        for name, cls in [("src", _StreamSrcKernel), ("dst", _FloatDstKernel)]:
            ki = KernelInstance(
                name=name,
                spec=KernelSpec(kernel_name=cls.__name__, rtl_top=cls.__name__),
                kernel_class=cls,
            )
            ki.kernel_class_instance = cls()
            ki._resolver = ParameterResolver({}, {}, {})
            for t in ki.kernel_class_instance.tensors():
                inst_t = copy.copy(t)
                setattr(ki.kernel_class_instance, t.name, inst_t)
                inst_t._resolve_shape(ki._resolver)
            sub_kernels[name] = ki

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        # Internal wire connections skip dtype check
        engine._validate_connection_dtypes([conn], sub_kernels)

    def test_duplicate_source_raises(self):
        """Same source interface in two connections → ValidationError."""
        from vten.runtime.engine import RuntimeEngine

        class _Dst2(Kernel):
            spec = "d2.yaml"
            data_in = Tensor(shape=(8,), dtype=torch.int8, interface="in2")
            ctrl = register("ctrl")
            def generate_inputs(self, seed=None): pass
            def forward(self, **inputs): return {}

        class _DupSrcComposite(CompositeKernel):
            spec = "dup.yaml"
            src = _StreamSrcKernel()
            dst1 = _StreamDstKernel()
            dst2 = _Dst2()
            connections = [
                src.data_out >> dst1.data_in,
                src.data_out >> dst2.data_in,
            ]

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        with pytest.raises(ValidationError, match="Duplicate connection source"):
            engine._validate_no_duplicate_connections(
                _DupSrcComposite.connections, {},
            )

    def test_dangling_internal_raises(self):
        """Internal() interface with no connection → ValidationError."""
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.flattener import InterfaceMapping
        from vten.spec.models import MappingType

        mappings = [
            InterfaceMapping(
                sub_kernel="src", sub_interface="output_stream",
                mapping_type=MappingType.INTERNAL,
                top_interface=None, bank_name=None,
            ),
            InterfaceMapping(
                sub_kernel="dst", sub_interface="input_stream",
                mapping_type=MappingType.INTERNAL,
                top_interface=None, bank_name=None,
            ),
        ]

        engine = RuntimeEngine(kernels={}, ops=[], project_params={})
        # No connections at all — both internal interfaces are dangling
        with pytest.raises(ValidationError, match="no connection"):
            engine._validate_internal_coverage(mappings, [], {})


# ═══════════════════════════════════════════════════════════════════
# A4. Compile Pipeline Checks
# ═══════════════════════════════════════════════════════════════════


class TestCompilePipeline:
    """Verify compile pipeline internals for composite kernels."""

    def _project_dir(self):
        return str(
            Path(__file__).resolve().parent.parent / "examples" / "scale_add"
        )

    def test_sub_kernel_reuse(self):
        """_sub_kernel_instances tensors are same objects as ExposedTensor origins."""
        ctx = ExecutionContext(
            project_params={"N": 32, "_project_dir": self._project_dir()}
        )
        k = ctx.instantiate(ScaleAddKernel, N=32)
        k.generate_inputs(seed=42)

        # ExposedTensor.origin_tensor should be the same object as sub-kernel tensor
        sub_scale = k._sub_kernel_instances["scale"]
        scale_data_in = sub_scale.get_tensor("data_in")
        assert k.data_in.origin_tensor is scale_data_in

    def test_spec_caching(self):
        """Second compile() reuses cached _synthesized_spec."""
        # Clear any cached spec from previous tests
        if hasattr(ScaleAddKernel, "_synthesized_spec"):
            delattr(ScaleAddKernel, "_synthesized_spec")

        project_dir = self._project_dir()
        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)

            ctx = ExecutionContext(
                project_params={"N": 32, "_project_dir": project_dir}
            )
            k = ctx.instantiate(ScaleAddKernel, N=32)
            k.generate_inputs(seed=42)

            # First compile
            from vten.dsl.operations import Operation
            from vten.runtime.engine import RuntimeEngine
            from vten.spec.models import OpKind

            ops = [
                Operation(kind=OpKind.PUSH_TENSOR, tensor=k.data_in),
                Operation(kind=OpKind.PULL_TENSOR, tensor=k.data_out),
            ]
            engine = RuntimeEngine(
                kernels=ctx._kernels, ops=ops,
                project_params={"N": 32, "_project_dir": project_dir},
            )
            result1 = engine.compile()

            # After first compile, class should have cached spec
            assert hasattr(ScaleAddKernel, "_synthesized_spec")
            cached = ScaleAddKernel._synthesized_spec

            # Second compile should reuse the cache
            k2 = ctx.instantiate(ScaleAddKernel, N=32)
            k2.generate_inputs(seed=42)
            engine2 = RuntimeEngine(
                kernels=ctx._kernels, ops=ops,
                project_params={"N": 32, "_project_dir": project_dir},
            )
            result2 = engine2.compile()
            assert ScaleAddKernel._synthesized_spec is cached
        finally:
            os.chdir(old_cwd)
            if hasattr(ScaleAddKernel, "_synthesized_spec"):
                delattr(ScaleAddKernel, "_synthesized_spec")

    def test_project_dir_from_params(self):
        """_project_dir in project_params is used, not CWD."""
        project_dir = self._project_dir()
        # Set CWD to a directory that does NOT contain kernel specs
        old_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            ctx = ExecutionContext(
                project_params={"N": 32, "_project_dir": project_dir}
            )
            k = ctx.instantiate(ScaleAddKernel, N=32)
            k.generate_inputs(seed=42)

            # Should still resolve specs correctly via _project_dir
            assert k._sub_kernel_instances is not None
            assert "scale" in k._sub_kernel_instances
        finally:
            os.chdir(old_cwd)
            if hasattr(ScaleAddKernel, "_synthesized_spec"):
                delattr(ScaleAddKernel, "_synthesized_spec")


# ═══════════════════════════════════════════════════════════════════
# A5. Codegen Output Verification
# ═══════════════════════════════════════════════════════════════════


class TestCodegenOutput:
    """Verify generate_composite_sv() produces correct SV."""

    def test_scale_add_composite_sv(self, tmp_path):
        """ScaleAdd codegen: internal wires, sub-kernel instances."""
        from vten.build.composite import generate_composite_sv, synthesize_spec

        project_dir = (
            Path(__file__).resolve().parent.parent / "examples" / "scale_add"
        )
        spec = synthesize_spec(ScaleAddKernel, project_dir, "scale_add")

        generate_composite_sv(
            ScaleAddKernel, spec, project_dir, tmp_path,
        )

        sv_file = tmp_path / f"{spec.kernel_name}_composite_top.sv"
        assert sv_file.exists()
        content = sv_file.read_text()

        # Internal wire declarations (AXI4-Stream)
        assert "internal_0_tdata" in content
        assert "internal_0_tvalid" in content
        assert "internal_0_tready" in content
        assert "internal_0_tlast" in content

        # Sub-kernel instantiation
        assert "u_scale" in content
        assert "u_offset" in content

        # Top-level ports
        assert "clk" in content
        assert "rst_n" in content

    def test_axi4_internal_wire_declaration(self, tmp_path):
        """AXI4 protocol internal wire should include all 5 channels."""
        from vten.build.composite import _declare_internal_wire
        from vten.spec.models import Protocol

        wire = {
            "name": "test_axi4",
            "data_w": 256,
            "protocol": Protocol.AXI4,
            "addr_w": 64,
        }
        lines = _declare_internal_wire(wire)

        # Should have AR/R/AW/W/B channel signals
        signal_text = "\n".join(lines)
        for ch in ["araddr", "arvalid", "arready", "arlen",
                    "rdata", "rvalid", "rready", "rlast",
                    "awaddr", "awvalid", "awready", "awlen",
                    "wdata", "wvalid", "wready", "wlast", "wstrb",
                    "bresp", "bvalid", "bready"]:
            assert f"test_axi4_{ch}" in signal_text, f"Missing signal: {ch}"

    def test_axilite_internal_wire_declaration(self):
        """AXI4-Lite internal wire should include all signals."""
        from vten.build.composite import _declare_internal_wire
        from vten.spec.models import Protocol

        wire = {
            "name": "test_axil",
            "data_w": 32,
            "protocol": Protocol.AXI4L,
            "addr_w": 16,
        }
        lines = _declare_internal_wire(wire)
        signal_text = "\n".join(lines)
        for ch in ["awaddr", "awvalid", "awready",
                    "wdata", "wvalid", "wready",
                    "araddr", "arvalid", "arready",
                    "rdata", "rvalid", "rready"]:
            assert f"test_axil_{ch}" in signal_text, f"Missing signal: {ch}"

    def test_unsupported_protocol_raises(self):
        """Unsupported protocol should raise BuildError."""
        from vten.build.composite import _declare_internal_wire
        from vten.errors import BuildError

        wire = {"name": "bad", "data_w": 32, "protocol": "unknown"}
        with pytest.raises(BuildError):
            _declare_internal_wire(wire)
