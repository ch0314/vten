"""Verification helpers — comparison and matching utilities.

Extracted from ExecutionContext. Pure/static functions only.

Integer comparisons are bit-exact by default. Callers may opt in to an
LSB tolerance (``lsb_tol``) for quantized outputs, and may attach a
:class:`~vten.spec.models.QuantSpec` so mismatch reports also show the
dequantized (real-domain) values. The QuantSpec NEVER loosens the
comparison by itself — tolerance only ever comes from ``lsb_tol``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vten.errors import VerificationError

if TYPE_CHECKING:
    from vten.spec.models import QuantSpec


def compare(
    hw_output: torch.Tensor,
    golden: torch.Tensor,
    *,
    lsb_tol: int = 0,
) -> bool:
    """Element-wise comparison with tolerance.

    Args:
        hw_output: Captured hardware output.
        golden: Expected reference values.
        lsb_tol: Optional integer-LSB tolerance. ``0`` (default) keeps the
            exact ``torch.equal`` fast path for integer tensors; ``> 0``
            passes when ``max |hw - golden| <= lsb_tol`` computed in int64
            (plain arithmetic distance — wrap-around is NOT small).
            Ignored for floating-point tensors.
    """
    if hw_output.shape != golden.shape:
        return False
    if golden.is_floating_point():
        return torch.allclose(hw_output.float(), golden.float(), atol=1e-6, rtol=1e-5)
    if lsb_tol <= 0:
        return torch.equal(hw_output, golden)
    return bool((_lsb_errors(hw_output, golden) <= lsb_tol).all())


def max_diff(hw_output: torch.Tensor, golden: torch.Tensor) -> float:
    """Maximum element-wise absolute difference."""
    a, b = hw_output.flatten().float(), golden.flatten().float()
    n = min(a.numel(), b.numel())
    return (a[:n] - b[:n]).abs().max().item()


def _lsb_errors(hw_output: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    """Per-element ``|hw - golden|`` in int64 (exact, no wrap-around)."""
    return (hw_output.to(torch.int64) - golden.to(torch.int64)).abs()


def check_match(
    tensor_name: str,
    hw_output: torch.Tensor,
    golden: torch.Tensor,
    *,
    shape: tuple | None = None,
    lsb_tol: int = 0,
    quant: QuantSpec | None = None,
) -> int:
    """Compare HW output against golden; raise VerificationError on mismatch.

    Args:
        tensor_name: Name used in the report.
        hw_output: Captured hardware output.
        golden: Expected reference values.
        shape: Logical shape for the report (defaults to ``hw_output.shape``).
        lsb_tol: Integer-LSB tolerance forwarded to :func:`compare`.
            ``0`` (default) keeps integer comparison bit-exact.
        quant: Optional QuantSpec for the tensor — REPORTING ONLY. When
            given, mismatch lines also show the dequantized values and the
            per-element LSB error. It never affects pass/fail.

    Returns:
        Maximum LSB error observed (``0`` for exact matches and for
        floating-point comparisons). Nonzero only when a nonzero
        ``lsb_tol`` allowed an inexact integer match to pass.

    Raises:
        VerificationError: On mismatch. For integer tensors the error
            carries ``max_lsb_err`` (max int64 ``|hw - golden|``).
    """
    is_int = not golden.is_floating_point() and not hw_output.is_floating_point()
    same_shape = hw_output.shape == golden.shape

    if compare(hw_output, golden, lsb_tol=lsb_tol):
        if is_int and lsb_tol > 0:
            return int(_lsb_errors(hw_output, golden).max().item())
        return 0

    md = max_diff(hw_output, golden)
    dtype_str = str(golden.dtype).replace("torch.", "")

    lsb_err: torch.Tensor | None = None
    max_lsb_err = 0
    if is_int and same_shape:
        lsb_err = _lsb_errors(hw_output, golden)
        max_lsb_err = int(lsb_err.max().item())
        diff_mask = lsb_err > max(lsb_tol, 0)
    else:
        diff_mask = hw_output != golden
    diff_indices = diff_mask.nonzero(as_tuple=False)
    n_diff = diff_indices.shape[0]

    deq_hw = deq_golden = None
    if quant is not None and lsb_err is not None:
        from vten.runtime.quant import dequantize

        deq_hw = dequantize(hw_output, quant)
        deq_golden = dequantize(golden, quant)

    detail_parts: list[str] = []
    _show = min(n_diff, 4)
    for i in range(_show):
        idx = tuple(diff_indices[i].tolist())
        idx_str = f"[{','.join(str(x) for x in idx)}]"
        if deq_hw is not None and deq_golden is not None and lsb_err is not None:
            detail_parts.append(
                f"  {idx_str}: hw={hw_output[idx].item()} "
                f"(≈{deq_hw[idx].item():.6g}), "
                f"golden={golden[idx].item()} "
                f"(≈{deq_golden[idx].item():.6g}), "
                f"lsb_err={lsb_err[idx].item()}"
            )
        else:
            detail_parts.append(
                f"  {idx_str}: expected={golden[idx].item()}, "
                f"actual={hw_output[idx].item()}"
            )
    if n_diff > _show:
        detail_parts.append(f"  ... and {n_diff - _show} more elements differ")

    effective_shape = shape or tuple(hw_output.shape)
    quant_str = ""
    if max_lsb_err and (lsb_tol > 0 or quant is not None):
        quant_str = f"max_lsb_err={max_lsb_err}, "
        if lsb_tol > 0:
            quant_str += f"lsb_tol={lsb_tol}, "
    msg = (
        f"Verification failed for tensor '{tensor_name}': "
        f"shape={effective_shape}, dtype={dtype_str}, max_diff={md}, "
        f"{quant_str}"
        f"{n_diff}/{hw_output.numel()} elements differ"
    )
    detail = "\n".join(detail_parts)
    if detail:
        msg += f"\n{detail}"

    raise VerificationError(
        msg,
        tensor=tensor_name,
        shape=effective_shape,
        max_diff=md,
        max_lsb_err=max_lsb_err,
    )
