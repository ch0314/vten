"""Quantization arithmetic helpers implementing QuantSpec semantics.

Vectorized torch implementations of the integer arithmetic a fixed-point
datapath performs, parameterized by :class:`vten.spec.models.QuantSpec`.
These are the building blocks for quantization-aware golden models: instead
of every kernel hand-rolling widen/clamp/wrap logic in ``forward()``, the
declared QuantSpec drives one shared, bit-exact implementation.

All integer arithmetic uses **int64 intermediates**. Inputs are widened to
int64 before any multiply/add/shift, so results are exact as long as the
true intermediate fits in 64 bits (guaranteed for the supported code widths:
products of two <=32-bit codes, sums of <=62-bit aligned codes).
``bits == 64`` is supported only for signed specs (int64 *is* the 64-bit
two's-complement domain); unsigned 64-bit codes raise NotImplementedError.

ROUNDING SEMANTICS (bit-exactness lives or dies here)
-----------------------------------------------------

All three modes are defined on the *arithmetic right shift* ``v >> sh``
(equivalently division by ``2**sh``):

``trunc``
    Floor — rounds toward **negative infinity**, matching Verilog's
    arithmetic shift ``>>>`` and Python's ``>>`` on negative ints.
    **NOT** C/Python ``int()`` truncation toward zero:
    ``-5 >>> 1 == -3`` (floor(-2.5)), not ``-2``.

``half_up``
    Add half an output LSB, then floor — the classic RTL
    ``(v + (1 << (sh-1))) >>> sh``. Ties round toward **positive
    infinity** for both signs: ``2.5 -> 3`` and ``-2.5 -> -2``.

``half_even``
    Convergent (banker's) rounding — nearest integer, ties to the even
    result: ``2.5 -> 2``, ``3.5 -> 4``, ``-2.5 -> -2``. Eliminates the
    systematic +0.5-LSB bias of ``half_up`` over long accumulations.

The same three names apply to float→code conversion in :func:`quantize`
(``trunc`` = floor, ``half_up`` = floor(y + 0.5), ``half_even`` = IEEE
round-to-nearest-even), computed in float64. :func:`requantize` between two
Q-format specs is a **pure-integer** shift path with no float detour — this
is the hardware-realistic post-MAC scaling and is exact by construction.
"""

from __future__ import annotations

import torch

from vten.spec.models import QuantSpec


def _as_int64(t: torch.Tensor) -> torch.Tensor:
    """Widen input codes to the int64 intermediate domain."""
    t = torch.as_tensor(t)
    if t.dtype.is_floating_point:
        raise TypeError(
            f"integer code tensor expected, got floating dtype {t.dtype}"
        )
    return t.to(torch.int64)


def _reject_affine(op: str, *specs: QuantSpec) -> None:
    for qs in specs:
        if qs.is_affine:
            raise NotImplementedError(
                f"{op}: affine QuantSpec (scale/zero_point) is not supported "
                f"— integer-exact arithmetic is defined for Q-format specs "
                f"only. Dequantize/requantize explicitly instead."
            )


def _round_shift(v: torch.Tensor, sh: int, mode: str) -> torch.Tensor:
    """Arithmetic right shift of int64 ``v`` by ``sh`` with rounding ``mode``.

    Exact-integer implementation of the three rounding modes (see module
    docstring). ``sh == 0`` is the identity. Requires ``sh >= 0``.
    """
    if sh == 0:
        return v
    if mode == "trunc":
        # Arithmetic shift right == floor division by 2**sh (Verilog >>>).
        return v >> sh
    half = 1 << (sh - 1)
    if mode == "half_up":
        # RTL: (v + (1 << (sh-1))) >>> sh — ties toward +inf.
        return (v + half) >> sh
    if mode == "half_even":
        q = v >> sh                    # floor quotient
        rem = v & ((1 << sh) - 1)      # remainder in [0, 2**sh), floor semantics
        round_up = (rem > half) | ((rem == half) & ((q & 1) == 1))
        return q + round_up.to(torch.int64)
    raise ValueError(f"unknown rounding mode {mode!r}")


def _round_float(y: torch.Tensor, mode: str) -> torch.Tensor:
    """Round float64 ``y`` to int64 codes per rounding ``mode``."""
    if mode == "trunc":
        r = torch.floor(y)
    elif mode == "half_up":
        r = torch.floor(y + 0.5)
    elif mode == "half_even":
        r = torch.round(y)  # IEEE round-to-nearest, ties-to-even on float64
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    return r.to(torch.int64)


def apply_overflow(t: torch.Tensor, qs: QuantSpec) -> torch.Tensor:
    """Fold int64 values into the code range of ``qs`` per its overflow mode.

    ``saturate`` clamps to ``[qs.qmin, qs.qmax]``; ``wrap`` keeps the low
    ``qs.bits`` bits and reinterprets them two's-complement when signed —
    exactly what an RTL register of that width does on overflow.

    Args:
        t: Integer tensor (any integer dtype; widened to int64).
        qs: Target QuantSpec (bits/signed/overflow are used).

    Returns:
        int64 tensor with every element in ``[qs.qmin, qs.qmax]``.
    """
    t = _as_int64(t)
    if qs.bits == 64:
        if qs.signed:
            # int64 intermediates are already exactly 64-bit two's-complement:
            # wrap is the identity and saturation cannot be detected.
            return t
        raise NotImplementedError(
            "bits=64 unsigned is not supported (int64 intermediates)"
        )
    if qs.overflow == "saturate":
        return t.clamp(qs.qmin, qs.qmax)
    # wrap: keep low `bits` bits, then sign-fold if the sign bit is set.
    mask = (1 << qs.bits) - 1
    u = t & mask
    if qs.signed:
        # Fold [2**(bits-1), 2**bits) down by 2**bits. Subtract the sign bit
        # twice so every intermediate fits in int64 even at bits=63.
        sign_bit = 1 << (qs.bits - 1)
        u = torch.where((u & sign_bit) != 0, (u - sign_bit) - sign_bit, u)
    return u


def quantize(x_float: torch.Tensor, qs: QuantSpec) -> torch.Tensor:
    """Real values → integer codes per ``qs`` (float64-mediated).

    Q-format: ``round(x * 2**frac_bits)``; affine:
    ``round(x / scale) + zero_point`` — rounding per ``qs.rounding``
    (see module docstring), then :func:`apply_overflow`.

    Note: this path necessarily goes through float64 — it is exact for the
    rounding decision on the given doubles, but is NOT the hardware
    requantization path. Use :func:`requantize` for integer-exact rescaling
    of existing codes.

    Args:
        x_float: Floating tensor of real values (computed in float64).
        qs: Target QuantSpec.

    Returns:
        int64 tensor of codes in ``[qs.qmin, qs.qmax]``.
    """
    x = torch.as_tensor(x_float).to(torch.float64)
    if qs.is_affine:
        y = x / qs.scale + qs.zero_point
    else:
        y = x * float(1 << qs.frac_bits)
    return apply_overflow(_round_float(y, qs.rounding), qs)


def dequantize(q: torch.Tensor, qs: QuantSpec) -> torch.Tensor:
    """Integer codes → real values per ``qs``.

    Q-format: ``q * 2**-frac_bits``; affine: ``(q - zero_point) * scale``.

    Args:
        q: Integer code tensor.
        qs: Source QuantSpec.

    Returns:
        float64 tensor of real values.
    """
    qf = _as_int64(q).to(torch.float64)
    if qs.is_affine:
        return (qf - qs.zero_point) * qs.scale
    return qf * (2.0 ** -qs.frac_bits)


def requantize(
    q: torch.Tensor, from_qs: QuantSpec, to_qs: QuantSpec
) -> torch.Tensor:
    """Re-scale codes from one QuantSpec to another — the hardware path.

    For two Q-format specs this is **exact integer arithmetic** (no float
    detour), mirroring what RTL does after a MAC: an arithmetic shift by
    ``from_qs.frac_bits - to_qs.frac_bits`` with rounding per
    ``to_qs.rounding`` (right shift; left shifts are exact), then overflow
    handling per ``to_qs.overflow``.

    If either spec is affine the conversion falls back to
    ``quantize(dequantize(q))`` through float64 — documented as NOT the
    bit-exact hardware path.

    Args:
        q: Codes valid under ``from_qs``.
        from_qs: Source QuantSpec.
        to_qs: Destination QuantSpec (rounding/overflow taken from here).

    Returns:
        int64 tensor of codes in ``[to_qs.qmin, to_qs.qmax]``.
    """
    if from_qs.is_affine or to_qs.is_affine:
        return quantize(dequantize(q, from_qs), to_qs)
    v = _as_int64(q)
    sh = from_qs.frac_bits - to_qs.frac_bits
    if sh >= 0:
        v = _round_shift(v, sh, to_qs.rounding)
    else:
        v = v << (-sh)  # gaining fractional bits is exact
    return apply_overflow(v, to_qs)


def qmul(
    a: torch.Tensor,
    a_qs: QuantSpec,
    b: torch.Tensor,
    b_qs: QuantSpec,
    out_qs: QuantSpec,
) -> torch.Tensor:
    """Fixed-point multiply: ``a * b`` requantized to ``out_qs``.

    Widens both operands to int64, multiplies exactly (the product carries
    ``a_qs.frac_bits + b_qs.frac_bits`` fractional bits), then shifts to
    ``out_qs.frac_bits`` with ``out_qs.rounding`` and applies
    ``out_qs.overflow`` — the standard post-multiplier rescale in RTL.
    Q-format specs only (affine raises NotImplementedError).

    Args:
        a, b: Code tensors valid under ``a_qs`` / ``b_qs``.
        a_qs, b_qs: Operand QuantSpecs.
        out_qs: Result QuantSpec.

    Returns:
        int64 tensor of codes in ``[out_qs.qmin, out_qs.qmax]``.
    """
    _reject_affine("qmul", a_qs, b_qs, out_qs)
    prod = _as_int64(a) * _as_int64(b)
    sh = a_qs.frac_bits + b_qs.frac_bits - out_qs.frac_bits
    if sh >= 0:
        v = _round_shift(prod, sh, out_qs.rounding)
    else:
        v = prod << (-sh)
    return apply_overflow(v, out_qs)


def qadd(
    a: torch.Tensor,
    a_qs: QuantSpec,
    b: torch.Tensor,
    b_qs: QuantSpec,
    out_qs: QuantSpec,
) -> torch.Tensor:
    """Fixed-point add: ``a + b`` requantized to ``out_qs``.

    Aligns both operands (exact left shift) to
    ``max(a_qs.frac_bits, b_qs.frac_bits)`` fractional bits, adds in int64,
    then shifts to ``out_qs.frac_bits`` with ``out_qs.rounding`` and applies
    ``out_qs.overflow``. Q-format specs only (affine raises
    NotImplementedError).

    Args:
        a, b: Code tensors valid under ``a_qs`` / ``b_qs``.
        a_qs, b_qs: Operand QuantSpecs.
        out_qs: Result QuantSpec.

    Returns:
        int64 tensor of codes in ``[out_qs.qmin, out_qs.qmax]``.
    """
    _reject_affine("qadd", a_qs, b_qs, out_qs)
    fb = max(a_qs.frac_bits, b_qs.frac_bits)
    acc = (_as_int64(a) << (fb - a_qs.frac_bits)) + (
        _as_int64(b) << (fb - b_qs.frac_bits)
    )
    sh = fb - out_qs.frac_bits
    if sh >= 0:
        v = _round_shift(acc, sh, out_qs.rounding)
    else:
        v = acc << (-sh)
    return apply_overflow(v, out_qs)
