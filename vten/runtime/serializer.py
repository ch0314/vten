"""Stage 3: Tensor Serialization.

StreamSerializer and MultiPortSerializer.

Spec reference: 02_runtime_engine.md §8
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from vten.spec.models import PackingScheme

# Element widths (bits) with a whole-byte, native-numpy representation. The
# vectorized fast paths only engage for these; everything else falls back to
# the exact per-element reference implementation.
_FAST_ELEMENT_WIDTHS = (8, 16, 32, 64)

if TYPE_CHECKING:
    from vten.spec.models import SplitSpec


def _flat_to_coords(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """C-contiguous flat index → tensor coordinates (row-major)."""
    coords: list[int] = []
    for dim in reversed(shape):
        coords.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(coords))


class StreamSerializer:
    """Serialize tensors to byte streams using PackingScheme."""

    def __init__(self, packing: PackingScheme) -> None:
        self.packing = packing

    def serialize(self, tensor_data: torch.Tensor) -> bytes:
        """Tensor → byte stream. Element order: C-contiguous (row-major).

        Uses a vectorized numpy path for byte-aligned element widths; falls
        back to the exact per-element reference implementation
        (:meth:`_serialize_slow`) for anything not provably handled.
        """
        fast = self._serialize_fast(tensor_data)
        if fast is not None:
            return fast
        return self._serialize_slow(tensor_data)

    def _serialize_slow(self, tensor_data: torch.Tensor) -> bytes:
        """Reference serialize: pure-Python per-beat packing loop."""
        flat = tensor_data.flatten()
        beats: list[bytes] = []
        for i in range(0, len(flat), self.packing.elements_per_beat):
            chunk = flat[i : i + self.packing.elements_per_beat]
            beat = self._pack_beat(chunk)
            beats.append(beat)
        return b"".join(beats)

    def _serialize_fast(self, tensor_data: torch.Tensor) -> bytes | None:
        """Vectorized serialize for byte-aligned element widths.

        Returns ``None`` (signalling the caller to use the slow path) unless
        the PackingScheme is standard-mode with an element width in
        {8,16,32,64} bits, a recognized bit/byte order, and a bus wide enough
        to hold ``elements_per_beat`` packed elements. Semantics match
        :meth:`_serialize_slow` byte-for-byte (including per-beat padding and
        a zero-padded final partial beat).
        """
        p = self.packing
        ew = p.element_width
        epb = p.elements_per_beat
        if p.mode != "standard":
            return None
        if ew not in _FAST_ELEMENT_WIDTHS or epb <= 0:
            return None
        if p.bit_order not in ("lsb_first", "msb_first"):
            return None
        if p.byte_order not in ("little", "big"):
            return None

        ew_bytes = ew // 8
        num_bytes = (p.bus_width + 7) // 8
        packed_bytes = epb * ew_bytes
        if num_bytes < packed_bytes:
            # Bus narrower than the packed elements: reference loop truncates
            # via int.to_bytes/overflow semantics — not modelled here.
            return None
        pad_bytes = num_bytes - packed_bytes

        try:
            arr = tensor_data.detach().cpu().flatten().numpy()
        except (RuntimeError, TypeError):
            return None
        if not np.issubdtype(arr.dtype, np.integer):
            return None

        n = int(arr.shape[0])
        rem = n % epb
        if rem != 0:
            # Zero-pad the flat data up to a whole beat; the padding elements
            # emit zero bytes, matching the reference's short-final-beat.
            arr = np.concatenate(
                [arr, np.zeros(epb - rem, dtype=arr.dtype)]
            )
        total = int(arr.shape[0])
        num_beats = total // epb

        # Mask each element to `ew` bits and lay it out little-endian in
        # `ew_bytes` bytes. astype(uint64) applies two's-complement wraparound
        # for negatives (== int(v) & mask); the &mask then trims to `ew` bits.
        mask = (1 << ew) - 1
        u64 = arr.astype(np.uint64) & np.uint64(mask)
        le = u64.astype(np.dtype(f"<u{ew_bytes}"))
        byte_view = le.view(np.uint8).reshape(num_beats, epb, ew_bytes)
        if p.bit_order == "msb_first":
            # element 0 occupies the top slot → reverse element order within
            # each beat when emitting little-endian bytes.
            byte_view = byte_view[:, ::-1, :]
        beats = byte_view.reshape(num_beats, packed_bytes)
        if pad_bytes:
            beats = np.concatenate(
                [beats, np.zeros((num_beats, pad_bytes), dtype=np.uint8)],
                axis=1,
            )
        if p.byte_order == "big":
            # to_bytes(num_bytes, 'big') reverses the whole beat's bytes.
            beats = beats[:, ::-1]
        return beats.tobytes()

    def _pack_beat(self, elements: torch.Tensor) -> bytes:
        beat_val = 0
        ew = self.packing.element_width
        mask = (1 << ew) - 1
        for idx, elem in enumerate(elements):
            raw = int(elem) & mask
            if self.packing.bit_order == "lsb_first":
                shift = idx * ew
            else:
                shift = (self.packing.elements_per_beat - 1 - idx) * ew
            beat_val |= raw << shift
        num_bytes = (self.packing.bus_width + 7) // 8
        return beat_val.to_bytes(num_bytes, byteorder=self.packing.byte_order)

    @staticmethod
    def _is_signed(dtype: torch.dtype | None) -> bool:
        """True if sign-extension applies (unsigned torch dtypes → False)."""
        _unsigned_dtypes = {torch.uint8}
        # torch.uint16/uint32/uint64 may not exist in older PyTorch versions
        for _name in ("uint16", "uint32", "uint64"):
            if hasattr(torch, _name):
                _unsigned_dtypes.add(getattr(torch, _name))
        return dtype not in _unsigned_dtypes if dtype is not None else True

    def deserialize(
        self,
        raw_bytes: bytes,
        num_elements: int,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Byte stream → Tensor. Element order: C-contiguous.

        Uses a vectorized numpy path for byte-aligned element widths; falls
        back to :meth:`_deserialize_slow` for anything not provably handled.
        """
        fast = self._deserialize_fast(raw_bytes, num_elements, shape, dtype)
        if fast is not None:
            return fast
        return self._deserialize_slow(raw_bytes, num_elements, shape, dtype)

    def _deserialize_fast(
        self,
        raw_bytes: bytes,
        num_elements: int,
        shape: tuple[int, ...] | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor | None:
        """Vectorized deserialize for byte-aligned element widths.

        Returns ``None`` to fall back to the reference loop unless the scheme
        is standard-mode, ew ∈ {8,16,32,64}, recognized bit/byte order, the
        bus holds the packed elements, and ``raw_bytes`` is a whole number of
        beats. Matches :meth:`_deserialize_slow` (incl. sign-extension for
        signed dtypes and ``num_elements`` truncation).
        """
        p = self.packing
        ew = p.element_width
        epb = p.elements_per_beat
        if p.mode != "standard":
            return None
        if ew not in _FAST_ELEMENT_WIDTHS or epb <= 0:
            return None
        if p.bit_order not in ("lsb_first", "msb_first"):
            return None
        if p.byte_order not in ("little", "big"):
            return None

        ew_bytes = ew // 8
        num_bytes = (p.bus_width + 7) // 8
        packed_bytes = epb * ew_bytes
        if num_bytes < packed_bytes:
            return None
        if num_bytes == 0 or len(raw_bytes) % num_bytes != 0:
            return None

        signed = self._is_signed(dtype)
        if ew == 64 and not signed:
            # uint64 values may exceed int64; slow path is exact here.
            return None

        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        num_beats = len(raw_bytes) // num_bytes
        beats = buf.reshape(num_beats, num_bytes)
        if p.byte_order == "big":
            beats = beats[:, ::-1]
        # Drop per-beat padding, split into per-element byte groups.
        elem_bytes = beats[:, :packed_bytes].reshape(num_beats, epb, ew_bytes)
        if p.bit_order == "msb_first":
            elem_bytes = elem_bytes[:, ::-1, :]
        flat = np.ascontiguousarray(elem_bytes).reshape(-1)
        if signed:
            vals = flat.view(np.dtype(f"<i{ew_bytes}"))
        else:
            vals = flat.view(np.dtype(f"<u{ew_bytes}"))
        vals = vals[:num_elements]
        # Widen to int64 so torch.from_numpy always has a supported dtype;
        # ew<=32 unsigned fits, ew==64 is signed-only here.
        vals = vals.astype(np.int64)
        tensor = torch.from_numpy(vals)
        out_dtype = dtype if dtype is not None else torch.int32
        tensor = tensor.to(out_dtype)
        if shape is not None:
            tensor = tensor.reshape(shape)
        return tensor

    def _deserialize_slow(
        self,
        raw_bytes: bytes,
        num_elements: int,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Reference deserialize: pure-Python per-element unpacking loop."""
        ew = self.packing.element_width
        epb = self.packing.elements_per_beat
        mask = (1 << ew) - 1
        bytes_per_beat = (self.packing.bus_width + 7) // 8

        signed = self._is_signed(dtype)

        elements: list[int] = []
        for beat_idx in range(0, len(raw_bytes), bytes_per_beat):
            beat_bytes = raw_bytes[beat_idx : beat_idx + bytes_per_beat]
            beat_val = int.from_bytes(beat_bytes, byteorder=self.packing.byte_order)
            for idx in range(epb):
                if len(elements) >= num_elements:
                    break
                if self.packing.bit_order == "lsb_first":
                    shift = idx * ew
                else:
                    shift = (epb - 1 - idx) * ew
                val = (beat_val >> shift) & mask
                # Sign-extend only for signed types
                if signed and val >= (1 << (ew - 1)):
                    val -= 1 << ew
                elements.append(val)

        out_dtype = dtype if dtype is not None else torch.int32
        tensor = torch.tensor(elements, dtype=out_dtype)
        if shape is not None:
            tensor = tensor.reshape(shape)
        return tensor

    def beat_index_to_coords(
        self, beat_index: int, shape: tuple[int, ...]
    ) -> list[tuple[int, ...]]:
        """beat_index → tensor coordinates of elements in that beat."""
        elem_start = beat_index * self.packing.elements_per_beat
        total_elems = math.prod(shape)
        coords: list[tuple[int, ...]] = []
        for elem_idx in range(
            elem_start,
            min(elem_start + self.packing.elements_per_beat, total_elems),
        ):
            coords.append(_flat_to_coords(elem_idx, shape))
        return coords


class CustomFieldSerializer:
    """Serialize/deserialize using custom field bit positions.

    Custom packing maps named fields to specific bit ranges within a beat.
    Each beat is a dict of {field_name: int_value}.
    """

    def __init__(self, packing: PackingScheme) -> None:
        if packing.mode != "custom" or not packing.custom_fields:
            raise ValueError("CustomFieldSerializer requires mode='custom' with fields")
        self.packing = packing
        self._fields = packing.custom_fields

    def serialize_beat(self, field_values: dict[str, int]) -> bytes:
        """Pack a single beat from field name→value dict."""
        beat_val = 0
        for cf in self._fields:
            value = field_values.get(cf.name, 0)
            lo, hi = cf.bits
            width = hi - lo + 1
            mask = (1 << width) - 1
            beat_val |= (int(value) & mask) << lo
        num_bytes = (self.packing.bus_width + 7) // 8
        return beat_val.to_bytes(num_bytes, byteorder=self.packing.byte_order)

    def serialize_beats(self, beats: list[dict[str, int]]) -> bytes:
        """Pack multiple beats into a byte stream."""
        return b"".join(self.serialize_beat(b) for b in beats)

    def deserialize_beat(self, raw_bytes: bytes) -> dict[str, int]:
        """Unpack a single beat into field name→value dict."""
        beat_val = int.from_bytes(raw_bytes, byteorder=self.packing.byte_order)
        result: dict[str, int] = {}
        for cf in self._fields:
            lo, hi = cf.bits
            width = hi - lo + 1
            mask = (1 << width) - 1
            result[cf.name] = (beat_val >> lo) & mask
        return result

    def deserialize_beats(self, raw_bytes: bytes, num_beats: int) -> list[dict[str, int]]:
        """Unpack multiple beats from a byte stream."""
        bytes_per_beat = (self.packing.bus_width + 7) // 8
        beats: list[dict[str, int]] = []
        for i in range(num_beats):
            chunk = raw_bytes[i * bytes_per_beat : (i + 1) * bytes_per_beat]
            beats.append(self.deserialize_beat(chunk))
        return beats


class MultiPortSerializer:
    """Split serialized data across multiple ports.

    Note: ``_interleave_split`` / ``reassemble`` remain byte-wise Python loops.
    They are left unvectorized on purpose — the round-robin distribution with
    a partial trailing unit and uneven per-port counts does not reduce to a
    clean numpy reshape, and these paths are not on the large-N hot path that
    ``StreamSerializer`` is. Out of scope for this change.
    """

    def split_tensor(
        self, serialized: bytes, split_spec: SplitSpec
    ) -> dict[str, bytes]:
        if split_spec.mode == "channel_interleave":
            return self._interleave_split(serialized, split_spec)
        elif split_spec.mode == "block_split":
            return self._block_split(serialized, split_spec)
        else:
            raise ValueError(f"Unknown split mode: {split_spec.mode}")

    def _interleave_split(
        self, data: bytes, spec: SplitSpec
    ) -> dict[str, bytes]:
        unit = spec.interleave.unit
        num_ports = len(spec.ports)
        result: dict[str, bytearray] = {p.name: bytearray() for p in spec.ports}
        for i in range(0, len(data), unit):
            port_idx = (i // unit) % num_ports
            result[spec.ports[port_idx].name].extend(data[i : i + unit])
        return {k: bytes(v) for k, v in result.items()}

    def _block_split(
        self, data: bytes, spec: SplitSpec
    ) -> dict[str, bytes]:
        num_ports = len(spec.ports)
        block_size = len(data) // num_ports
        result: dict[str, bytes] = {}
        for i, port in enumerate(spec.ports):
            start = i * block_size
            end = start + block_size if i < num_ports - 1 else len(data)
            result[port.name] = data[start:end]
        return result

    @staticmethod
    def reassemble(port_data: dict[str, bytes], interleave_unit: int) -> bytes:
        """Reverse channel_interleave: round-robin reassembly."""
        ports = list(port_data.values())
        n_ports = len(ports)
        result = bytearray()
        offsets = [0] * n_ports
        while any(offsets[i] < len(ports[i]) for i in range(n_ports)):
            for i in range(n_ports):
                chunk = ports[i][offsets[i] : offsets[i] + interleave_unit]
                result.extend(chunk)
                offsets[i] += interleave_unit
        return bytes(result)


# ── Split/block helpers (used by engine Stage 4 + shm_packer) ──


def parse_split_spec(raw):
    """Parse a raw dict or SplitSpec into a SplitSpec dataclass."""
    from vten.spec.models import InterleaveSpec, PortDef, SplitSpec

    if isinstance(raw, SplitSpec):
        return raw
    ports = [
        PortDef(name=p["name"], base_addr=p.get("base_addr", 0))
        for p in raw.get("ports", [])
    ]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)


def block_split_data(
    serialized: bytes | None,
    flat_names: list[str],
    serialized_size: int,
) -> dict[str, bytes]:
    """Block-split serialized data (or allocate empty) across port names."""
    n = len(flat_names)
    if serialized is not None:
        data = serialized
        chunk_size = len(data) // n
        remainder = len(data) % n
        result = {}
        offset = 0
        for i, fname in enumerate(flat_names):
            sz = chunk_size + (1 if i < remainder else 0)
            result[fname] = data[offset : offset + sz]
            offset += sz
        return result
    else:
        per_elem_size = serialized_size // n
        return {fname: bytes(per_elem_size) for fname in flat_names}
