"""Runtime pipeline tests — Stage 1: Parameter Resolution.

Spec reference: 02_runtime_engine.md §6, 00_data_models.md §7.4
NPU 3D patterns: npu_3d_analysis.md §3

Tests the ParameterResolver class: hierarchical scope chain,
expression substitution, and error handling.
"""

from __future__ import annotations

import math

import pytest
import torch

from vten.errors import ParameterResolutionError


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def resolver_cls():
    """Import ParameterResolver lazily."""
    from vten.runtime.resolver import ParameterResolver
    return ParameterResolver


# ═══════════════════════════════════════════════════════════════════
# §6.1 — Hierarchical Scope Chain (Priority 1 > 2 > 3)
# ═══════════════════════════════════════════════════════════════════


class TestScopeChain:
    """Priority: runtime_params > kernel_params > project_params."""

    def test_project_params_lowest_priority(self, resolver_cls):
        r = resolver_cls(
            project_params={"C": 64},
            kernel_params={},
            runtime_params={},
        )
        assert r.resolve("${C}") == 64

    def test_kernel_params_override_project(self, resolver_cls):
        r = resolver_cls(
            project_params={"C": 64},
            kernel_params={"C": 128},
            runtime_params={},
        )
        assert r.resolve("${C}") == 128

    def test_runtime_params_override_all(self, resolver_cls):
        r = resolver_cls(
            project_params={"C": 64},
            kernel_params={"C": 128},
            runtime_params={"C": 32},
        )
        assert r.resolve("${C}") == 32

    def test_mixed_scopes_independent_params(self, resolver_cls):
        """Each param can come from a different scope."""
        r = resolver_cls(
            project_params={"BUS_WIDTH": 256},
            kernel_params={"STRIDE": 1},
            runtime_params={"C": 64, "D": 4},
        )
        assert r.resolve("${C}") == 64
        assert r.resolve("${BUS_WIDTH}") == 256
        assert r.resolve("${STRIDE}") == 1
        assert r.resolve("${D}") == 4

    def test_npu_3d_typical_params(self, resolver_cls):
        """NPU 3D layer config: runtime overrides for shape params."""
        r = resolver_cls(
            project_params={"BUS_WIDTH": 256},
            kernel_params={
                "IN_DEPTH": "${IN_DEPTH}", "IN_HEIGHT": "${IN_HEIGHT}",
                "IN_WIDTH": "${IN_WIDTH}", "IN_CH": "${IN_CH}",
                "OUT_CH": "${OUT_CH}", "KERNEL_SIZE": 3,
            },
            runtime_params={
                "IN_DEPTH": 4, "IN_HEIGHT": 8, "IN_WIDTH": 8,
                "IN_CH": 32, "OUT_CH": 64,
            },
        )
        assert r.resolve("${IN_CH}") == 32
        assert r.resolve("${OUT_CH}") == 64
        assert r.resolve("${KERNEL_SIZE}") == 3


# ═══════════════════════════════════════════════════════════════════
# §6.3 — Expression Evaluation
# ═══════════════════════════════════════════════════════════════════


class TestExpressionEval:
    """resolve() handles ${...} substitution + arithmetic."""

    def test_integer_passthrough(self, resolver_cls):
        """Non-string values pass through unchanged."""
        r = resolver_cls({}, {}, {})
        assert r.resolve(42) == 42
        assert r.resolve(0) == 0

    def test_string_without_placeholder(self, resolver_cls):
        """Strings without ${} are returned as-is or evaluated."""
        r = resolver_cls({}, {}, {})
        # Plain numeric string should evaluate to int
        assert r.resolve("42") == 42

    def test_simple_substitution(self, resolver_cls):
        r = resolver_cls({}, {}, {"N": 16})
        assert r.resolve("${N}") == 16

    def test_arithmetic_addition(self, resolver_cls):
        r = resolver_cls({}, {}, {"D": 8, "K": 3})
        assert r.resolve("${D}+${K}") == 11

    def test_arithmetic_subtraction(self, resolver_cls):
        r = resolver_cls({}, {}, {"D": 8, "K": 3})
        assert r.resolve("${D}-${K}") == 5

    def test_arithmetic_multiplication(self, resolver_cls):
        r = resolver_cls({}, {}, {"C": 32, "TILE": 4})
        assert r.resolve("${C}*${TILE}") == 128

    def test_floor_division(self, resolver_cls):
        r = resolver_cls({}, {}, {"C": 64, "TILE": 32})
        assert r.resolve("${C}//${TILE}") == 2

    def test_complex_expression(self, resolver_cls):
        """Output dim formula: (D - K) // stride + 1."""
        r = resolver_cls({}, {}, {"D": 8, "K": 3, "STRIDE": 1})
        result = r.resolve("(${D}-${K})//${STRIDE}+1")
        assert result == 6

    def test_npu_3d_channel_pkt(self, resolver_cls):
        """NPU channel packing: ceil(CH / Ti) = (CH + Ti - 1) // Ti."""
        r = resolver_cls({}, {}, {"IN_CH": 48, "Ti": 32})
        result = r.resolve("(${IN_CH}+${Ti}-1)//${Ti}")
        assert result == 2  # ceil(48/32) = 2

    def test_npu_3d_output_dim_stride2(self, resolver_cls):
        """Downsample: out = ceil(in / 2) = (in + 1) // 2."""
        r = resolver_cls({}, {}, {"IN_HEIGHT": 8})
        result = r.resolve("(${IN_HEIGHT}+1)//2")
        assert result == 4

    def test_npu_3d_upsample_dim(self, resolver_cls):
        """Upsample (transpose conv): out = in * 2."""
        r = resolver_cls({}, {}, {"IN_DEPTH": 4})
        result = r.resolve("${IN_DEPTH}*2")
        assert result == 8

    def test_multiple_same_param(self, resolver_cls):
        """Same parameter used multiple times in one expression."""
        r = resolver_cls({}, {}, {"N": 4})
        assert r.resolve("${N}*${N}") == 16


# ═══════════════════════════════════════════════════════════════════
# §6.3 — namespace attribute
# ═══════════════════════════════════════════════════════════════════


class TestNamespace:
    """resolver.namespace exposes merged params for KernelInstance."""

    def test_namespace_contains_all_params(self, resolver_cls):
        r = resolver_cls(
            project_params={"A": 1},
            kernel_params={"B": 2},
            runtime_params={"C": 3},
        )
        assert r.namespace["A"] == 1
        assert r.namespace["B"] == 2
        assert r.namespace["C"] == 3

    def test_namespace_override_order(self, resolver_cls):
        r = resolver_cls(
            project_params={"X": 10},
            kernel_params={"X": 20},
            runtime_params={"X": 30},
        )
        assert r.namespace["X"] == 30


# ═══════════════════════════════════════════════════════════════════
# §16.2 V5 — ParameterResolutionError
# ═══════════════════════════════════════════════════════════════════


class TestParameterErrors:
    """Unresolved parameters raise ParameterResolutionError."""

    def test_unresolved_single_param(self, resolver_cls):
        r = resolver_cls({}, {}, {})
        with pytest.raises(ParameterResolutionError, match="MISSING"):
            r.resolve("${MISSING}")

    def test_unresolved_in_expression(self, resolver_cls):
        r = resolver_cls({}, {}, {"A": 1})
        with pytest.raises(ParameterResolutionError, match="B"):
            r.resolve("${A}+${B}")

    def test_error_lists_available_params(self, resolver_cls):
        r = resolver_cls({}, {}, {"X": 1, "Y": 2})
        with pytest.raises(ParameterResolutionError, match="Available"):
            r.resolve("${Z}")

    def test_kernel_param_referencing_unset_runtime(self, resolver_cls):
        """kernel_spec has ${C} but no scope provides C's value."""
        r = resolver_cls(
            project_params={},
            kernel_params={"OUT_DIM": "${C}"},
            runtime_params={},
        )
        with pytest.raises(ParameterResolutionError):
            r.resolve("${OUT_DIM}")


# ═══════════════════════════════════════════════════════════════════
# §6.3 — safe_eval (only arithmetic allowed)
# ═══════════════════════════════════════════════════════════════════


class TestSafeEval:
    """Only arithmetic operators allowed — no builtins, imports, etc."""

    def test_modulo(self, resolver_cls):
        r = resolver_cls({}, {}, {"N": 10})
        assert r.resolve("${N}%3") == 1

    def test_power_rejected_or_evaluated(self, resolver_cls):
        """Depending on safe_eval implementation, ** may or may not work.
        This test documents behavior — at minimum should not crash."""
        r = resolver_cls({}, {}, {"N": 2})
        try:
            result = r.resolve("${N}**3")
            assert result == 8
        except (ParameterResolutionError, ValueError):
            pass  # Also acceptable if ** is rejected

    def test_negative_result(self, resolver_cls):
        r = resolver_cls({}, {}, {"A": 3, "B": 5})
        assert r.resolve("${A}-${B}") == -2


# ═══════════════════════════════════════════════════════════════════
# Integration: Tensor shape resolution pattern
# ═══════════════════════════════════════════════════════════════════


class TestTensorShapeResolution:
    """Simulates how Stage 2 uses the resolver for tensor shapes."""

    def test_resolve_shape_tuple(self, resolver_cls):
        """Resolve each dimension in a tensor shape tuple."""
        r = resolver_cls({}, {}, {
            "N": 1, "C": 64, "D": 4, "H": 8, "W": 8,
        })
        shape = ("${N}", "${C}", "${D}", "${H}", "${W}")
        resolved = tuple(r.resolve(dim) for dim in shape)
        assert resolved == (1, 64, 4, 8, 8)
        assert math.prod(resolved) == 1 * 64 * 4 * 8 * 8

    def test_npu_3d_ifm_tiled_shape(self, resolver_cls):
        """NPU IFM tiled shape: (D, C_pkt, H, W, Ti)."""
        r = resolver_cls({}, {}, {
            "IN_DEPTH": 4, "IN_CH": 48, "Ti": 32,
            "IN_HEIGHT": 8, "IN_WIDTH": 8,
        })
        shape = (
            "${IN_DEPTH}",
            "(${IN_CH}+${Ti}-1)//${Ti}",
            "${IN_HEIGHT}",
            "${IN_WIDTH}",
            "${Ti}",
        )
        resolved = tuple(r.resolve(dim) for dim in shape)
        assert resolved == (4, 2, 8, 8, 32)

    def test_npu_3d_weight_shape(self, resolver_cls):
        """NPU weight tiled shape with channel packing."""
        r = resolver_cls({}, {}, {
            "IN_CH": 32, "OUT_CH": 64, "Ti": 32, "To": 32,
            "KERNEL_SIZE": 3,
        })
        shape = (
            "(${IN_CH}+${Ti}-1)//${Ti}",
            "${Ti}",
            "(${OUT_CH}+${To}-1)//${To}",
            "${To}",
            "${KERNEL_SIZE}*${KERNEL_SIZE}*${KERNEL_SIZE}",
        )
        resolved = tuple(r.resolve(dim) for dim in shape)
        # IN_CH_pkt=1, Ti=32, OUT_CH_pkt=2, To=32, K^3=27
        assert resolved == (1, 32, 2, 32, 27)

    def test_bias_shape(self, resolver_cls):
        """Bias: padded to To boundary."""
        r = resolver_cls({}, {}, {"OUT_CH": 48, "To": 32})
        shape = ("(${OUT_CH}+${To}-1)//${To}*${To}",)
        resolved = tuple(r.resolve(dim) for dim in shape)
        assert resolved == (64,)  # ceil(48/32)*32 = 64

    def test_mixed_literal_and_param_shape(self, resolver_cls):
        """Some dims are literals, some are parametric."""
        r = resolver_cls({}, {}, {"SIZE": 1024})
        shape = ("${SIZE}",)
        resolved = tuple(r.resolve(dim) for dim in shape)
        assert resolved == (1024,)


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary and corner cases for resolver."""

    def test_zero_value_param(self, resolver_cls):
        r = resolver_cls({}, {}, {"IS_CONCAT": 0})
        assert r.resolve("${IS_CONCAT}") == 0

    def test_large_address_value(self, resolver_cls):
        """64-bit addresses common in NPU DDR/HBM."""
        r = resolver_cls({}, {}, {"BASE": 0x8000_0000})
        assert r.resolve("${BASE}") == 0x8000_0000

    def test_empty_params(self, resolver_cls):
        """All scopes empty — only literal values work."""
        r = resolver_cls({}, {}, {})
        assert r.resolve(100) == 100

    def test_param_value_is_string_integer(self, resolver_cls):
        """kernel_spec params may have string values like '3'."""
        r = resolver_cls({}, {"K": "3"}, {})
        # Should resolve ${K} to integer 3
        assert r.resolve("${K}") == 3


# ═══════════════════════════════════════════════════════════════════
# §6.5 — build_params tier
# ═══════════════════════════════════════════════════════════════════


class TestBuildParamsTier:
    """build_params sit between project_params and kernel_params."""

    def test_build_params_available(self, resolver_cls):
        """build_params values accessible in namespace."""
        r = resolver_cls(
            {}, {}, {},
            build_params={"Ti": 32, "To": 32},
        )
        assert r.resolve("${Ti}") == 32
        assert r.resolve("${To}") == 32

    def test_build_params_below_kernel_params(self, resolver_cls):
        """kernel_params (Tier 3) override build_params (Tier 2)."""
        r = resolver_cls(
            {},
            {"Ti": 64},  # kernel_params
            {},
            build_params={"Ti": 32},
        )
        assert r.resolve("${Ti}") == 64

    def test_build_params_above_project_params(self, resolver_cls):
        """build_params (Tier 2) override project_params (Tier 1)."""
        r = resolver_cls(
            {"Ti": 16},  # project_params
            {},
            {},
            build_params={"Ti": 32},
        )
        assert r.resolve("${Ti}") == 32

    def test_build_params_override_warning(self, resolver_cls, caplog):
        """Test override warning when runtime overrides build_param."""
        import logging
        with caplog.at_level(logging.WARNING, logger="vten.runtime.resolver"):
            r = resolver_cls(
                {},
                {},
                {"Ti": 64},  # runtime override
                build_params={"Ti": 32},
            )
        assert "overrides build_param" in caplog.text
        # Override still works
        assert r.resolve("${Ti}") == 64

    def test_build_params_none_backward_compat(self, resolver_cls):
        """build_params=None → backward compatible 3-tier."""
        r = resolver_cls({"A": 1}, {"B": 2}, {"C": 3})
        assert r.resolve("${A}") == 1
        assert r.resolve("${B}") == 2
        assert r.resolve("${C}") == 3


# ═══════════════════════════════════════════════════════════════════
# §6.6 — runtime_param_specs defaults
# ═══════════════════════════════════════════════════════════════════


class TestDefaultParamsDefaults:
    """default_params (Kernel.default_params) use setdefault (Tier 3)."""

    def test_default_params_applied(self, resolver_cls):
        """default_params fills in missing params."""
        r = resolver_cls(
            {}, {}, {},
            default_params={"in_depth": 4},
        )
        assert r.resolve("${in_depth}") == 4

    def test_default_params_does_not_override_kernel_params(self, resolver_cls):
        """default_params doesn't override kernel_params (Tier 4)."""
        r = resolver_cls(
            {},
            {"in_depth": 8},  # kernel_params already has it
            {},
            default_params={"in_depth": 4},
        )
        assert r.resolve("${in_depth}") == 8

    def test_default_params_scalar(self, resolver_cls):
        """Scalar default_params value used as setdefault."""
        r = resolver_cls(
            {}, {}, {},
            default_params={"in_width": 4},
        )
        assert r.resolve("${in_width}") == 4

    def test_test_override_wins_over_default_params(self, resolver_cls):
        """Tier 5 test override beats default_params."""
        r = resolver_cls(
            {}, {},
            {"in_depth": 16},  # test override
            default_params={"in_depth": 4},
        )
        assert r.resolve("${in_depth}") == 16
