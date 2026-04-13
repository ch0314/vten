"""Output tensor reading and deserialization.

Extracted from ExecutionContext to keep context.py focused on orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from vten.spec.models import Direction

if TYPE_CHECKING:
    from vten.kernel.tensor import Tensor
    from vten.runtime.engine import CompiledResult
    from vten.runtime.kernel_view import ExposedTensor, FlattenedKernelView


def read_output_tensors(
    compiled: CompiledResult,
    backend_result: object,
    *,
    is_hw: bool = False,
    get_buffer_object: Callable[[int], object] | None = None,
) -> dict[str, Tensor]:
    """Deserialize DEV_TO_HOST tensors from backend result.

    Args:
        compiled: The compiled result from RuntimeEngine.
        backend_result: The backend execution result.
        is_hw: True if the backend is hardware (XRT), enables BO binding.
        get_buffer_object: Callable(buffer_id) → BO, required when is_hw=True.

    Returns:
        Dict mapping tensor name → deserialized Tensor.
    """
    from vten.kernel.tensor import Tensor as TensorCls
    from vten.runtime.engine import RuntimeEngine
    from vten.runtime.serializer import StreamSerializer

    view = compiled.flattened_view

    # Fast path: CPU backend provides forward() tensors directly
    forward_tensors = getattr(backend_result, '_forward_tensors', None)

    output_tensors: dict[str, TensorCls] = {}
    for name, exposed in view.exposed_tensors.items():
        if exposed.direction != Direction.DEV_TO_HOST:
            continue

        origin = exposed.origin_tensor
        try:
            iface = view.top_spec.get_interface(exposed.top_interface)
        except KeyError:
            continue
        if iface.packing is None:
            continue

        t = TensorCls(
            shape=origin._resolved_shape or origin.shape,
            dtype=origin.dtype,
            interface=origin.interface,
            direction=Direction.DEV_TO_HOST,
        )
        t.name = name
        t._resolved_shape = origin._resolved_shape
        t._element_count = origin._element_count

        if forward_tensors is not None and name in forward_tensors:
            # CPU backend fast path: use forward() tensor directly (no serialize/deserialize)
            t.data = RuntimeEngine._apply_unlayout(view, exposed, forward_tensors[name])
        else:
            raw_bytes = read_tensor_bytes(name, exposed, compiled, backend_result)
            if raw_bytes:
                serializer = StreamSerializer(iface.packing)
                hw_tensor = serializer.deserialize(
                    raw_bytes,
                    origin._element_count,
                    origin._resolved_shape,
                    dtype=origin.dtype,
                )
                t.data = RuntimeEngine._apply_unlayout(view, exposed, hw_tensor)

        if is_hw and get_buffer_object is not None:
            buffer_id = compiled.buffer_ids.get(name)
            if buffer_id is not None:
                bo = get_buffer_object(buffer_id)
                if bo is not None:
                    deserialize_fn = make_deserialize_fn(view, exposed)
                    bo_size = (bo.size() if hasattr(bo, "size")
                               else getattr(exposed, "_serialized_size", 0))
                    t._bind_bo(bo, bo_size, deserialize_fn)

        output_tensors[name] = t
    return output_tensors


def make_deserialize_fn(
    view: FlattenedKernelView,
    exposed: ExposedTensor,
) -> Callable[[bytes], torch.Tensor] | None:
    """Create a bytes → torch.Tensor deserialize+unlayout closure for BO binding."""
    from vten.runtime.engine import RuntimeEngine
    from vten.runtime.serializer import StreamSerializer

    try:
        iface = view.top_spec.get_interface(exposed.top_interface)
    except (KeyError, AttributeError):
        return None
    if iface.packing is None:
        return None

    packing = iface.packing
    origin = exposed.origin_tensor
    element_count = origin._element_count
    shape = origin._resolved_shape
    dtype = origin.dtype

    def _deserialize(raw: bytes) -> torch.Tensor:
        serializer = StreamSerializer(packing)
        hw_tensor = serializer.deserialize(raw, element_count, shape, dtype=dtype)
        return RuntimeEngine._apply_unlayout(view, exposed, hw_tensor)

    return _deserialize


def read_tensor_bytes(
    name: str,
    exposed: ExposedTensor,
    compiled: CompiledResult,
    backend_result: object,
    buffer_prefix: str = "",
) -> bytes:
    """Read raw bytes for a tensor, reassembling array/chunk buffers."""
    prefixed = f"{buffer_prefix}{name}"
    chunk_0_key = f"{prefixed}:chunk_0"
    is_chunked = any(k.startswith(chunk_0_key) for k in compiled.buffer_ids)

    if is_chunked:
        return read_all_chunk_bytes(
            name, exposed, compiled, backend_result,
            buffer_prefix=buffer_prefix,
        )

    if exposed._port_buffers:
        parts = {}
        for port_name in exposed._port_buffers:
            key = f"{prefixed}:{port_name}"
            bid = compiled.buffer_ids.get(key)
            if bid is None:
                continue
            data = backend_result.read_buffer(bid)
            if data:
                parts[port_name] = data
        if exposed._port_mode == "channel_interleave" and parts:
            from vten.runtime.serializer import MultiPortSerializer
            return MultiPortSerializer.reassemble(parts, exposed._interleave_unit)
        return b"".join(parts.values())

    buffer_id = compiled.buffer_ids[prefixed]
    return backend_result.read_buffer(buffer_id)


def read_all_chunk_bytes(
    name: str,
    exposed: ExposedTensor,
    compiled: CompiledResult,
    backend_result: object,
    buffer_prefix: str = "",
) -> bytes:
    """Read and concatenate all chunk buffers for a chunked tensor."""
    prefixed = f"{buffer_prefix}{name}"
    parts: list[bytes] = []
    ci = 0
    while True:
        if exposed._port_buffers:
            chunk_parts: list[bytes] = []
            for fname in exposed._port_buffers:
                key = f"{prefixed}:chunk_{ci}:{fname}"
                bid = compiled.buffer_ids.get(key)
                if bid is None:
                    return b"".join(parts)
                data = backend_result.read_buffer(bid)
                if data:
                    chunk_parts.append(data)
            parts.extend(chunk_parts)
        else:
            key = f"{prefixed}:chunk_{ci}"
            bid = compiled.buffer_ids.get(key)
            if bid is None:
                break
            data = backend_result.read_buffer(bid)
            if data:
                parts.append(data)
        ci += 1
    return b"".join(parts)


def read_chunk_bytes(
    name: str,
    exposed: ExposedTensor,
    compiled: CompiledResult,
    backend_result: object,
    chunk_index: int,
    buffer_prefix: str = "",
) -> bytes:
    """Read raw bytes for a single chunk of a chunked tensor."""
    prefixed = f"{buffer_prefix}{name}"
    if exposed._port_buffers:
        parts: list[bytes] = []
        for fname in exposed._port_buffers:
            key = f"{prefixed}:chunk_{chunk_index}:{fname}"
            bid = compiled.buffer_ids.get(key)
            if bid is None:
                continue
            data = backend_result.read_buffer(bid)
            if data:
                parts.append(data)
        return b"".join(parts)
    key = f"{prefixed}:chunk_{chunk_index}"
    bid = compiled.buffer_ids[key]
    return backend_result.read_buffer(bid)
