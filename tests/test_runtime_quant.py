"""Differential tests for vten.runtime.quant — QuantSpec arithmetic helpers.

Every vectorized torch op is checked element-for-element against a slow,
pure-Python scalar reference implementation over a grid of QuantSpecs
(mirroring the fast-vs-slow differential style of test_runtime_serializer):

    bits {4, 8, 12, 16, 24, 32} x signed {T, F} x frac_bits {0, bits//2}
    x overflow {saturate, wrap} x rounding {trunc, half_up, half_even}
    = 144 spec combinations

with data = seeded random + adversarial edges (qmin/qmax, +-1 around the
saturation boundary, values just past the wrap point, exact rounding ties
at both quotient parities, zero).

Also pins the rounding-mode contracts explicitly:
  - trunc     == floor / Verilog ``>>>`` (toward -inf, NOT toward zero)
  - half_up   == ``(v + (1 << (sh-1))) >>> sh`` (ties toward +inf)
  - half_even == convergent/banker's rounding
"""

from __future__ import annotations

import itertools
import math
import random

import pytest
import torch

from vten.runtime import quant
from vten.spec.models import QuantSpec


# ═══════════════════════════════════════════════════════════════════
# Scalar pure-Python reference implementations
# ═══════════════════════════════════════════════════════════════════


def ref_overflow(v: int, qs: QuantSpec) -> int:
    if qs.overflow == "saturate":
        return min(max(v, qs.qmin), qs.qmax)
    u = v % (1 << qs.bits)
    if qs.signed and u >= (1 << (qs.bits - 1)):
        u -= 1 << qs.bits
    return u


def ref_round_shift(v: int, sh: int, mode: str) -> int:
    if sh == 0:
        return v
    if mode == "trunc":
        return v >> sh  # Python >> on ints is arithmetic/floor, like Verilog >>>
    half = 1 << (sh - 1)
    if mode == "half_up":
        return (v + half) >> sh
    # half_even
    q, rem = v >> sh, v & ((1 << sh) - 1)
    if rem > half or (rem == half and q % 2 != 0):
        return q + 1
    return q


def ref_round_float(y: float, mode: str) -> int:
    if mode == "trunc":
        return math.floor(y)
    if mode == "half_up":
        return math.floor(y + 0.5)
    # half_even on the same double
    f = math.floor(y)
    r = y - f
    if r > 0.5:
        return f + 1
    if r < 0.5:
        return f
    return f if f % 2 == 0 else f + 1


def ref_quantize(x: float, qs: QuantSpec) -> int:
    if qs.is_affine:
        y = x / qs.scale + qs.zero_point
    else:
        y = x * float(1 << qs.frac_bits)
    return ref_overflow(ref_round_float(y, qs.rounding), qs)


def ref_dequantize(q: int, qs: QuantSpec) -> float:
    if qs.is_affine:
        return (q - qs.zero_point) * qs.scale
    return q * (2.0 ** -qs.frac_bits)


def ref_requantize(q: int, from_qs: QuantSpec, to_qs: QuantSpec) -> int:
    if from_qs.is_affine or to_qs.is_affine:
        return ref_quantize(ref_dequantize(q, from_qs), to_qs)
    sh = from_qs.frac_bits - to_qs.frac_bits
    v = ref_round_shift(q, sh, to_qs.rounding) if sh >= 0 else q << (-sh)
    return ref_overflow(v, to_qs)


def ref_qmul(
    a: int, a_qs: QuantSpec, b: int, b_qs: QuantSpec, out_qs: QuantSpec
) -> int:
    p = a * b
    sh = a_qs.frac_bits + b_qs.frac_bits - out_qs.frac_bits
    v = ref_round_shift(p, sh, out_qs.rounding) if sh >= 0 else p << (-sh)
    return ref_overflow(v, out_qs)


def ref_qadd(
    a: int, a_qs: QuantSpec, b: int, b_qs: QuantSpec, out_qs: QuantSpec
) -> int:
    fb = max(a_qs.frac_bits, b_qs.frac_bits)
    acc = (a << (fb - a_qs.frac_bits)) + (b << (fb - b_qs.frac_bits))
    sh = fb - out_qs.frac_bits
    v = ref_round_shift(acc, sh, out_qs.rounding) if sh >= 0 else acc << (-sh)
    return ref_overflow(v, out_qs)


# ═══════════════════════════════════════════════════════════════════
# Grid + data generation
# ═══════════════════════════════════════════════════════════════════

_GRID = list(itertools.product(
    (4, 8, 12, 16, 24, 32),          # bits
    (True, False),                    # signed
    ("fb0", "fbhalf"),                # frac_bits: 0 or bits // 2
    ("saturate", "wrap"),             # overflow
    ("trunc", "half_up", "half_even"),  # rounding
))  # 6 * 2 * 2 * 2 * 3 = 144

_GRID_IDS = [
    f"b{b}-{'s' if s else 'u'}-{fb}-{ov}-{rd}"
    for b, s, fb, ov, rd in _GRID
]


def _mk_qs(bits, signed, fb_mode, overflow, rounding) -> QuantSpec:
    fb = 0 if fb_mode == "fb0" else bits // 2
    return QuantSpec(
        bits=bits, signed=signed, frac_bits=fb,
        overflow=overflow, rounding=rounding,
    )


def _adversarial_ints(qs: QuantSpec) -> list[int]:
    """Edge codes: range boundaries, +-1 around saturation, past wrap point."""
    span = 1 << qs.bits
    vals = {
        0, 1, -1,
        qs.qmin, qs.qmax,
        qs.qmin - 1, qs.qmin + 1,       # just past / inside saturation
        qs.qmax - 1, qs.qmax + 1,
        qs.qmax + span, qs.qmin - span,  # just past the wrap point
        qs.qmax + span + 1, qs.qmin - span - 1,
        2 * qs.qmax + 1, 2 * qs.qmin - 1,
        span, -span, span - 1, -(span - 1),
    }
    return sorted(vals)


def _random_ints(qs: QuantSpec, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    lo, hi = -(1 << (qs.bits + 2)), (1 << (qs.bits + 2))
    return [rng.randint(lo, hi) for _ in range(n)]


def _codes_in_range(qs: QuantSpec, n: int, seed: int) -> list[int]:
    """Valid codes under qs: boundaries + seeded randoms within [qmin, qmax]."""
    rng = random.Random(seed)
    vals = {qs.qmin, qs.qmax, qs.qmin + 1, qs.qmax - 1, 0, 1}
    if qs.signed:
        vals.add(-1)
    vals.update(rng.randint(qs.qmin, qs.qmax) for _ in range(n))
    return sorted(vals)


def _tie_codes(qs: QuantSpec, sh: int) -> list[int]:
    """Exact rounding ties for a right shift by sh: v = (k << sh) + half,
    at even and odd quotient parities, both signs, clipped into qs range."""
    if sh <= 0:
        return []
    half = 1 << (sh - 1)
    ties = [(k << sh) + half for k in range(-4, 4)]
    return [v for v in ties if qs.qmin <= v <= qs.qmax]


def _assert_equal(out: torch.Tensor, ref: list[int], ctx: str) -> None:
    assert out.dtype == torch.int64
    got = out.tolist()
    assert got == ref, (
        f"{ctx}: vectorized != scalar reference\n"
        f"  got: {got}\n  ref: {ref}"
    )


# ═══════════════════════════════════════════════════════════════════
# Differential grid: vectorized torch == scalar Python reference
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("bits,signed,fb_mode,overflow,rounding",
                         _GRID, ids=_GRID_IDS)
class TestQuantDifferential:

    def test_apply_overflow(self, bits, signed, fb_mode, overflow, rounding):
        qs = _mk_qs(bits, signed, fb_mode, overflow, rounding)
        data = _adversarial_ints(qs) + _random_ints(qs, 64, seed=bits * 7 + 1)
        out = quant.apply_overflow(torch.tensor(data, dtype=torch.int64), qs)
        ref = [ref_overflow(v, qs) for v in data]
        _assert_equal(out, ref, f"apply_overflow {qs}")
        assert int(out.min()) >= qs.qmin
        assert int(out.max()) <= qs.qmax

    def test_quantize_dequantize(self, bits, signed, fb_mode, overflow, rounding):
        qs = _mk_qs(bits, signed, fb_mode, overflow, rounding)
        lsb = qs.lsb
        rng = random.Random(bits * 13 + qs.frac_bits)
        # Exact rounding ties (x = (k + 0.5) * lsb) at both parities, range
        # boundaries +- 1 code, zero, and uniform randoms past the range.
        vals = [(k + 0.5) * lsb for k in range(-4, 4)]
        vals += [0.0, lsb, -lsb,
                 qs.qmin * lsb, qs.qmax * lsb,
                 (qs.qmin - 1) * lsb, (qs.qmax + 1) * lsb,
                 (qs.qmin - 0.5) * lsb, (qs.qmax + 0.5) * lsb]
        vals += [rng.uniform((qs.qmin - 4) * lsb, (qs.qmax + 4) * lsb)
                 for _ in range(64)]
        x = torch.tensor(vals, dtype=torch.float64)

        q = quant.quantize(x, qs)
        ref_q = [ref_quantize(v, qs) for v in vals]
        _assert_equal(q, ref_q, f"quantize {qs}")

        d = quant.dequantize(q, qs)
        ref_d = [ref_dequantize(c, qs) for c in ref_q]
        assert d.dtype == torch.float64
        assert d.tolist() == ref_d, f"dequantize {qs}"

    def test_requantize(self, bits, signed, fb_mode, overflow, rounding):
        from_qs = _mk_qs(bits, signed, fb_mode, overflow, rounding)
        fb = from_qs.frac_bits
        # Destination variants: drop all frac bits (post-MAC rescale), gain
        # frac bits (exact left shift), and narrow the width (saturation/wrap
        # stress) — always with the same rounding/overflow under test.
        to_variants = {
            (bits, 0),
            (bits, min(fb + 3, bits - 1)),
            (max(2, bits // 2), 0),
        }
        for to_bits, to_fb in sorted(to_variants):
            to_qs = QuantSpec(
                bits=to_bits, signed=signed, frac_bits=to_fb,
                overflow=overflow, rounding=rounding,
            )
            sh = fb - to_fb
            data = (_codes_in_range(from_qs, 48, seed=bits + to_bits)
                    + _tie_codes(from_qs, sh))
            out = quant.requantize(
                torch.tensor(data, dtype=torch.int64), from_qs, to_qs
            )
            ref = [ref_requantize(v, from_qs, to_qs) for v in data]
            _assert_equal(out, ref, f"requantize {from_qs} -> {to_qs}")

    def test_qmul(self, bits, signed, fb_mode, overflow, rounding):
        a_qs = _mk_qs(bits, signed, fb_mode, overflow, rounding)
        b_qs = QuantSpec(bits=bits, signed=signed, frac_bits=0,
                         overflow=overflow, rounding=rounding)
        # Cap operand magnitude at 2**31 so |a*b| < 2**62 stays exact in the
        # int64 intermediate (documented helper contract).
        cap_lo = max(a_qs.qmin, -(1 << 31))
        cap_hi = min(a_qs.qmax, (1 << 31) - 1)
        rng = random.Random(bits * 17 + a_qs.frac_bits)
        edge = [v for v in (cap_lo, cap_hi, 0, 1, -1, cap_lo + 1, cap_hi - 1)
                if a_qs.qmin <= v <= a_qs.qmax]
        pairs = list(itertools.product(edge, edge))
        pairs += [(rng.randint(cap_lo, cap_hi), rng.randint(cap_lo, cap_hi))
                  for _ in range(48)]
        a = torch.tensor([p[0] for p in pairs], dtype=torch.int64)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.int64)
        for out_qs in (
            a_qs,
            QuantSpec(bits=bits, signed=signed, frac_bits=0,
                      overflow=overflow, rounding=rounding),
            QuantSpec(bits=min(2 * bits, 48), signed=signed,
                      frac_bits=a_qs.frac_bits, overflow=overflow,
                      rounding=rounding),
        ):
            out = quant.qmul(a, a_qs, b, b_qs, out_qs)
            ref = [ref_qmul(pa, a_qs, pb, b_qs, out_qs) for pa, pb in pairs]
            _assert_equal(out, ref, f"qmul {a_qs} x {b_qs} -> {out_qs}")

    def test_qadd(self, bits, signed, fb_mode, overflow, rounding):
        a_qs = _mk_qs(bits, signed, fb_mode, overflow, rounding)
        b_qs = QuantSpec(bits=bits, signed=signed, frac_bits=0,
                         overflow=overflow, rounding=rounding)
        a_vals = _codes_in_range(a_qs, 24, seed=bits * 3)
        b_vals = _codes_in_range(b_qs, 24, seed=bits * 5)
        n = min(len(a_vals), len(b_vals))
        rng = random.Random(bits)
        rng.shuffle(a_vals)
        rng.shuffle(b_vals)
        a = torch.tensor(a_vals[:n], dtype=torch.int64)
        b = torch.tensor(b_vals[:n], dtype=torch.int64)
        for out_qs in (
            a_qs,
            QuantSpec(bits=bits, signed=signed, frac_bits=0,
                      overflow=overflow, rounding=rounding),
            QuantSpec(bits=min(2 * bits, 48), signed=signed,
                      frac_bits=min(a_qs.frac_bits + 1, bits),
                      overflow=overflow, rounding=rounding),
        ):
            out = quant.qadd(a, a_qs, b, b_qs, out_qs)
            ref = [ref_qadd(int(pa), a_qs, int(pb), b_qs, out_qs)
                   for pa, pb in zip(a.tolist(), b.tolist())]
            _assert_equal(out, ref, f"qadd {a_qs} + {b_qs} -> {out_qs}")


# ═══════════════════════════════════════════════════════════════════
# Pinned rounding semantics (the contract, not just self-consistency)
# ═══════════════════════════════════════════════════════════════════


def _q(bits=16, signed=True, fb=0, overflow="saturate", rounding="trunc"):
    return QuantSpec(bits=bits, signed=signed, frac_bits=fb,
                     overflow=overflow, rounding=rounding)


def _requant_codes(codes, from_fb, to_fb, rounding):
    out = quant.requantize(
        torch.tensor(codes, dtype=torch.int64),
        _q(fb=from_fb),
        _q(fb=to_fb, rounding=rounding),
    )
    return out.tolist()


class TestRoundingSemantics:

    def test_trunc_is_floor_not_toward_zero(self):
        """trunc == arithmetic shift right (Verilog >>>): -2.5 -> -3."""
        # Q1 codes: -5 = -2.5, -1 = -0.5, 5 = 2.5
        assert _requant_codes([-5, -1, 5], 1, 0, "trunc") == [-3, -1, 2]

    def test_trunc_matches_verilog_asr(self):
        codes = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 7]
        assert _requant_codes(codes, 2, 0, "trunc") == [c >> 2 for c in codes]

    def test_half_up_ties_toward_plus_inf(self):
        """RTL (v + (1 << (sh-1))) >>> sh: 2.5 -> 3 AND -2.5 -> -2."""
        # Q1 codes: 5 = 2.5, 3 = 1.5, -3 = -1.5, -5 = -2.5
        assert _requant_codes([5, 3, -3, -5], 1, 0, "half_up") == [3, 2, -1, -2]

    def test_half_even_convergent(self):
        # Q1 codes: 3 = 1.5, 5 = 2.5, 7 = 3.5, -3 = -1.5, -5 = -2.5
        assert _requant_codes([3, 5, 7, -3, -5], 1, 0, "half_even") == \
            [2, 2, 4, -2, -2]

    def test_half_even_non_ties_round_nearest(self):
        # Q2 codes: 5 = 1.25 -> 1, 7 = 1.75 -> 2, -5 = -1.25 -> -1
        assert _requant_codes([5, 7, -5, -7], 2, 0, "half_even") == [1, 2, -1, -2]

    def test_left_shift_gaining_frac_bits_is_exact(self):
        assert _requant_codes([3, -3], 0, 2, "trunc") == [12, -12]

    def test_quantize_float_rounding_modes(self):
        x = torch.tensor([2.5, -2.5, 1.5, -1.5, 0.4, -0.4], dtype=torch.float64)
        assert quant.quantize(x, _q(rounding="trunc")).tolist() == \
            [2, -3, 1, -2, 0, -1]
        assert quant.quantize(x, _q(rounding="half_up")).tolist() == \
            [3, -2, 2, -1, 0, 0]
        assert quant.quantize(x, _q(rounding="half_even")).tolist() == \
            [2, -2, 2, -2, 0, 0]


class TestOverflowSemantics:

    def test_wrap_twos_complement_int8(self):
        qs = _q(bits=8, overflow="wrap")
        t = torch.tensor([128, 129, 255, 256, -129, -130, 127, -128])
        assert quant.apply_overflow(t, qs).tolist() == \
            [-128, -127, -1, 0, 127, 126, 127, -128]

    def test_wrap_unsigned_8(self):
        qs = _q(bits=8, signed=False, overflow="wrap")
        t = torch.tensor([256, 257, -1, -2, 255, 0])
        assert quant.apply_overflow(t, qs).tolist() == [0, 1, 255, 254, 255, 0]

    def test_saturate_signed_8(self):
        qs = _q(bits=8)
        t = torch.tensor([128, 1000, -129, -1000, 127, -128, 0])
        assert quant.apply_overflow(t, qs).tolist() == \
            [127, 127, -128, -128, 127, -128, 0]

    def test_saturate_unsigned_8(self):
        qs = _q(bits=8, signed=False)
        t = torch.tensor([256, -1, 255, 0])
        assert quant.apply_overflow(t, qs).tolist() == [255, 0, 255, 0]


class TestAffine:

    def test_quantize_dequantize_round_trip(self):
        qs = QuantSpec(bits=8, signed=False, scale=0.5, zero_point=10,
                       rounding="half_even")
        x = torch.tensor([0.0, 1.0, -5.0, 100.0, 122.5, -6.0],
                         dtype=torch.float64)
        q = quant.quantize(x, qs)
        assert q.tolist() == [10, 12, 0, 210, 255, 0]  # clamped at both ends
        d = quant.dequantize(q, qs)
        assert d.tolist() == [0.0, 1.0, -5.0, 100.0, 122.5, -5.0]

    def test_requantize_affine_float_detour(self):
        from_qs = QuantSpec(bits=8, signed=False, scale=0.5, zero_point=0)
        to_qs = QuantSpec(bits=8, signed=False, scale=1.0, zero_point=0)
        q = torch.tensor([0, 2, 4, 255])
        out = quant.requantize(q, from_qs, to_qs)
        ref = [ref_requantize(v, from_qs, to_qs) for v in q.tolist()]
        assert out.tolist() == ref == [0, 1, 2, 127]

    def test_qmul_affine_raises(self):
        aff = QuantSpec(bits=8, signed=True, scale=0.1)
        qf = _q(bits=8)
        t = torch.tensor([1])
        with pytest.raises(NotImplementedError, match="qmul.*affine"):
            quant.qmul(t, aff, t, qf, qf)
        with pytest.raises(NotImplementedError, match="qmul.*affine"):
            quant.qmul(t, qf, t, qf, aff)

    def test_qadd_affine_raises(self):
        aff = QuantSpec(bits=8, signed=True, scale=0.1)
        qf = _q(bits=8)
        t = torch.tensor([1])
        with pytest.raises(NotImplementedError, match="qadd.*affine"):
            quant.qadd(t, aff, t, qf, qf)


class TestEdgesAndErrors:

    def test_bits64_signed_identity(self):
        qs = QuantSpec(bits=64, signed=True, overflow="wrap")
        t = torch.tensor([torch.iinfo(torch.int64).min, -1, 0,
                          torch.iinfo(torch.int64).max])
        assert quant.apply_overflow(t, qs).tolist() == t.tolist()

    def test_bits64_unsigned_raises(self):
        qs = QuantSpec(bits=64, signed=False)
        with pytest.raises(NotImplementedError, match="bits=64 unsigned"):
            quant.apply_overflow(torch.tensor([1]), qs)

    def test_float_input_to_integer_op_raises(self):
        with pytest.raises(TypeError, match="integer code tensor expected"):
            quant.apply_overflow(torch.tensor([1.0]), _q())
        with pytest.raises(TypeError, match="integer code tensor expected"):
            quant.requantize(torch.tensor([1.5]), _q(fb=1), _q())

    def test_narrow_int_dtypes_widened(self):
        qs = _q(bits=8, overflow="wrap")
        out = quant.apply_overflow(torch.tensor([-1, 127], dtype=torch.int8), qs)
        assert out.dtype == torch.int64
        assert out.tolist() == [-1, 127]

    def test_shape_preserved(self):
        qs = _q(bits=8)
        t = torch.arange(12, dtype=torch.int64).reshape(3, 4) * 100
        out = quant.apply_overflow(t, qs)
        assert out.shape == (3, 4)

    def test_qmul_post_mac_rescale_example(self):
        """int8 Q4 x int8 Q4 -> int8 Q4 with half_up: the classic RTL path."""
        a_qs = b_qs = _q(bits=8, fb=4, rounding="half_up")
        out_qs = _q(bits=8, fb=4, rounding="half_up")
        # 1.5 * 1.5 = 2.25 -> Q4 code 36; 24*24=576, 576 >> 4 (half_up) = 36
        a = torch.tensor([24])  # 1.5 in Q4
        assert quant.qmul(a, a_qs, a, b_qs, out_qs).tolist() == [36]
        d = quant.dequantize(torch.tensor([36]), out_qs)
        assert d.tolist() == [2.25]

    def test_qadd_mixed_frac_alignment(self):
        """Q4 + Q0 aligns the integer operand up before adding."""
        a_qs = _q(bits=16, fb=4)
        b_qs = _q(bits=16, fb=0)
        out_qs = _q(bits=16, fb=4)
        a = torch.tensor([24])   # 1.5 in Q4
        b = torch.tensor([2])    # 2.0 in Q0
        out = quant.qadd(a, a_qs, b, b_qs, out_qs)
        assert out.tolist() == [56]  # 3.5 in Q4
