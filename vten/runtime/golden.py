"""Shared golden computation for verification.

Extracted from ExecutionContext._run_forward() and _compute_auto_golden()
so that both CLI (vten run) and inference API share identical logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vten.runtime.flattener import FlattenedKernelView

logger = logging.getLogger(__name__)


def run_forward(kernel_inst: object) -> dict[str, torch.Tensor]:
    """Run forward() on a kernel instance, handling Composite vs Simple.

    CompositeKernel: forward() with no args (auto-chain with layout).
    Simple Kernel: collect H2D tensor data, apply layout, forward(**inputs).

    For device-resident inputs in inference chains, falls back to
    tensor.golden (the golden data propagated from previous layer).
    """
    from vten.kernel.composite import CompositeKernel

    if isinstance(kernel_inst, CompositeKernel):
        return kernel_inst.forward()

    # Simple kernel: collect H2D inputs with layout
    inputs: dict[str, torch.Tensor] = {}
    for tensor in kernel_inst.tensors():
        data = tensor.data
        # Fallback to golden for device-resident tensors in inference chain
        if data is None:
            data = tensor.golden
        if data is None:
            continue
        direction = getattr(tensor, "direction", None)
        if direction is None or direction.value == "host_to_dev":
            layout_fn = getattr(kernel_inst, f"layout_{tensor.name}", None)
            if layout_fn is not None and callable(layout_fn):
                inputs[tensor.name] = layout_fn(data)
            else:
                inputs[tensor.name] = data

    return kernel_inst.forward(**inputs)


def compute_golden_outputs(
    kernel_inst: object,
    view: FlattenedKernelView,
    *,
    fwd_result: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute logical golden outputs from kernel's forward().

    Returns dict of tensor_name → golden torch.Tensor (logical format).

    Steps per output tensor:
      1. forward() → physical golden
      2. Format conversion (packing round-trip) for dtype alignment
      3. unlayout → logical golden

    Args:
        kernel_inst: The kernel instance to compute golden for.
        view: FlattenedKernelView from compiled result.
        fwd_result: Pre-computed forward() result. If None, calls run_forward().
    """
    from vten.runtime.engine import RuntimeEngine
    from vten.runtime.serializer import StreamSerializer
    from vten.spec.models import Direction

    if fwd_result is None:
        fwd_result = run_forward(kernel_inst)

    outputs: dict[str, torch.Tensor] = {}
    for name, exposed in view.exposed_tensors.items():
        if exposed.direction != Direction.DEV_TO_HOST:
            continue
        if name not in fwd_result:
            continue

        golden_phys = fwd_result[name].flatten()

        # Format conversion: packing round-trip for dtype alignment
        # The round-trip simulates HW precision loss (element_width truncation).
        # Deserialize with the *physical* dtype (forward output dtype, typically
        # uint8) to avoid sign-extension of unsigned values 128-255, then cast
        # to the logical target dtype.
        origin = exposed.origin_tensor
        target_dtype = origin.dtype
        if golden_phys.dtype != target_dtype:
            phys_dtype = golden_phys.dtype  # preserve physical dtype for round-trip
            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
                if iface.packing is not None:
                    serializer = StreamSerializer(iface.packing)
                    raw = serializer.serialize(golden_phys)
                    golden_phys = serializer.deserialize(
                        raw, origin._element_count,
                        origin._resolved_shape,
                        dtype=phys_dtype,
                    ).flatten()
            except (KeyError, AttributeError):
                pass

            # Cast to logical dtype after unsigned round-trip
            if golden_phys.dtype != target_dtype:
                golden_phys = golden_phys.to(target_dtype)

        # Apply unlayout
        golden_logical = RuntimeEngine._apply_unlayout(view, exposed, golden_phys)
        outputs[name] = golden_logical

    return outputs
