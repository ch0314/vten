"""Verification helpers — comparison, matching, and chunk utilities.

Extracted from ExecutionContext. Pure/static functions only; stateful
orchestration (run_deferred_verifications) stays in ExecutionContext.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vten.errors import VerificationError

if TYPE_CHECKING:
    from vten.runtime.flattener import ExposedTensor


def compare(hw_output: torch.Tensor, golden: torch.Tensor) -> bool:
    """Element-wise comparison with tolerance."""
    if hw_output.shape != golden.shape:
        return False
    if golden.is_floating_point():
        return torch.allclose(hw_output.float(), golden.float(), atol=1e-6, rtol=1e-5)
    return torch.equal(hw_output, golden)


def max_diff(hw_output: torch.Tensor, golden: torch.Tensor) -> float:
    """Maximum element-wise absolute difference."""
    a, b = hw_output.flatten().float(), golden.flatten().float()
    n = min(a.numel(), b.numel())
    return (a[:n] - b[:n]).abs().max().item()


def check_match(
    tensor_name: str,
    hw_output: torch.Tensor,
    golden: torch.Tensor,
    *,
    shape: tuple | None = None,
) -> None:
    """Compare HW output against golden; raise VerificationError on mismatch."""
    if compare(hw_output, golden):
        return

    md = max_diff(hw_output, golden)
    dtype_str = str(golden.dtype).replace("torch.", "")
    diff_mask = hw_output != golden
    diff_indices = diff_mask.nonzero(as_tuple=False)
    n_diff = diff_indices.shape[0]

    detail_parts: list[str] = []
    _show = min(n_diff, 4)
    for i in range(_show):
        idx = tuple(diff_indices[i].tolist())
        idx_str = f"[{','.join(str(x) for x in idx)}]"
        detail_parts.append(
            f"  {idx_str}: expected={golden[idx].item()}, "
            f"actual={hw_output[idx].item()}"
        )
    if n_diff > _show:
        detail_parts.append(f"  ... and {n_diff - _show} more elements differ")

    effective_shape = shape or tuple(hw_output.shape)
    msg = (
        f"Verification failed for tensor '{tensor_name}': "
        f"shape={effective_shape}, dtype={dtype_str}, max_diff={md}, "
        f"{n_diff}/{hw_output.numel()} elements differ"
    )
    detail = "\n".join(detail_parts)
    if detail:
        msg += f"\n{detail}"

    raise VerificationError(msg, tensor=tensor_name, shape=effective_shape, max_diff=md)


def chunk_element_info(
    exposed: ExposedTensor,
    chunk_index: int,
    chunk_total: int,
    chunks_spec: int | list[int] | None,
) -> tuple[int, tuple[int, ...]]:
    """Compute element count and shape for a single chunk."""
    total_elems = exposed.origin_tensor._element_count
    if isinstance(chunks_spec, list):
        chunk_elems = chunks_spec[chunk_index]
    else:
        chunk_elems = total_elems // chunk_total
    return chunk_elems, (chunk_elems,)
