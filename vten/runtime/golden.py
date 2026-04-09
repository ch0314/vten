"""Shared golden computation for verification.

Extracted from ExecutionContext._run_forward() and _compute_auto_golden()
so that both CLI (vten run) and inference API share identical logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vten.runtime.kernel_view import FlattenedKernelView

logger = logging.getLogger(__name__)


def _count_chunks(name: str, buffer_ids: dict[str, int]) -> int:
    """Count chunk indices for a tensor in buffer_ids."""
    n = 0
    while any(k.startswith(f"{name}:chunk_{n}") for k in buffer_ids):
        n += 1
    return n


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
    buffer_ids: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute logical golden outputs from kernel's forward().

    Returns dict of tensor_name → golden torch.Tensor (logical format).

    Steps per output tensor:
      1. forward() → physical golden
      2. Byte reorder for chunked array tensors (stream-first → chunk-port)
      3. Format conversion (packing round-trip) for dtype alignment
      4. unlayout → logical golden

    Args:
        kernel_inst: The kernel instance to compute golden for.
        view: FlattenedKernelView from compiled result.
        fwd_result: Pre-computed forward() result. If None, calls run_forward().
        buffer_ids: Compiled buffer ID map. Needed to detect chunk count
            for chunked array tensors (byte reorder).
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

        # Format conversion: forward() may return physical packed bytes
        # (e.g. uint8 stream with 21-bit packed elements) while the tensor
        # dtype is int32. Deserialize the raw bytes to get logical values.
        # Skip serialize — forward() output IS the packed format already.
        origin = exposed.origin_tensor
        target_dtype = origin.dtype
        if golden_phys.dtype != target_dtype:
            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
                if iface.packing is not None:
                    serializer = StreamSerializer(iface.packing)
                    raw_arr = golden_phys.numpy()

                    # Chunked array tensors: forward() returns bytes in
                    # stream-first order (port0_all, port1_all, ...) but
                    # read_output_tensors reassembles in chunk-port order
                    # (chunk0_port0, chunk0_port1, ..., chunk1_port0, ...).
                    # Rearrange to match the SHM read order.
                    raw_arr = _reorder_for_chunks(
                        raw_arr, name, exposed, iface.packing, buffer_ids,
                    )

                    raw = raw_arr.tobytes()
                    golden_phys = serializer.deserialize(
                        raw, origin._element_count,
                        origin._resolved_shape,
                        dtype=target_dtype,
                    ).flatten()
            except (KeyError, AttributeError):
                pass

            if golden_phys.dtype != target_dtype:
                golden_phys = golden_phys.to(target_dtype)

        # Apply unlayout
        golden_logical = RuntimeEngine._apply_unlayout(view, exposed, golden_phys)
        outputs[name] = golden_logical

    return outputs


def _reorder_for_chunks(
    raw_arr,
    name: str,
    exposed,
    packing,
    buffer_ids: dict[str, int] | None,
):
    """Reorder forward() bytes from stream-first to chunk-port order.

    forward() produces: port0_all_beats | port1_all_beats | ...
    SHM reads produce:  chunk0_port0 | chunk0_port1 | ... | chunk1_port0 | ...

    Only applies when the tensor has both array ports AND chunks > 1.
    Returns the array unchanged if no reordering is needed.
    """
    import numpy as np

    if not exposed._port_buffers or buffer_ids is None:
        return raw_arr

    n_chunks = _count_chunks(name, buffer_ids)
    if n_chunks <= 1:
        return raw_arr

    n_ports = len(exposed._port_buffers)
    bytes_per_beat = (packing.bus_width + 7) // 8
    total_bytes = raw_arr.size

    if total_bytes == 0 or total_bytes % (n_ports * bytes_per_beat) != 0:
        return raw_arr

    total_beats = total_bytes // (n_ports * bytes_per_beat)
    if total_beats % n_chunks != 0:
        return raw_arr

    beats_per_chunk = total_beats // n_chunks

    # stream-first: (n_ports, total_beats_per_port, bytes_per_beat)
    arr = raw_arr.reshape(n_ports, total_beats, bytes_per_beat)
    # split beats into chunks: (n_ports, n_chunks, beats_per_chunk, bytes_per_beat)
    arr = arr.reshape(n_ports, n_chunks, beats_per_chunk, bytes_per_beat)
    # transpose to chunk-first: (n_chunks, n_ports, beats_per_chunk, bytes_per_beat)
    arr = arr.transpose(1, 0, 2, 3)
    return arr.reshape(-1)
