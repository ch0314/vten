"""Tensor descriptor class.

Spec reference: 00_data_models.md §2

v2: shape/dtype are declared shape. If layout_{name}() exists on the kernel,
the shape is logical and the framework auto-calls layout before serialization.
Otherwise the shape is physical (HW/DDR) and data is serialized as-is.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from typing import Any


class Tensor:
    """Kernel tensor declaration. Lifecycle spans instantiate() → compile()."""

    def __init__(
        self,
        shape: tuple[str | int, ...],
        dtype: torch.dtype,
        interface: str,
        direction: Any | None = None,
    ) -> None:
        # Declaration time (Kernel class body)
        self.shape = shape          # logical (algorithmic) shape
        self.dtype = dtype          # logical (algorithmic) dtype
        self.interface = interface
        self.direction = direction  # None → inferred from protocol/role at Stage 0
        self.name: str = ""

        # instantiate() time (eager resolution)
        self._resolved_shape: tuple[int, ...] | None = None
        self._element_count: int = 0

        # Runtime data — logical if layout_{name}() exists, physical otherwise
        self.data: torch.Tensor | None = None
        self._address: int | None = None

        # Device state (inference mode)
        self._bo: Any = None                          # xrt.bo instance
        self._bo_size: int = 0                        # serialized byte count
        self._deserialize_fn: Callable[[bytes], torch.Tensor] | None = None

        # Verification state
        self.golden: torch.Tensor | None = None   # behavioral model golden (logical)
        self.verified: bool = False                # has verify() been called?
        self.max_diff: float = 0.0                 # max abs diff from golden

    @property
    def resolved_shape(self) -> tuple[int, ...]:
        """Resolved logical shape (read-only)."""
        if self._resolved_shape is None:
            raise RuntimeError("shape not resolved")
        return self._resolved_shape

    # ── Shape resolution ──

    def _resolve_shape(self, resolver: Any) -> None:
        """Resolve parametric shape dimensions using resolver.resolve()."""
        resolved = tuple(resolver.resolve(dim) for dim in self.shape)
        self._resolved_shape = resolved
        self._element_count = math.prod(resolved)

    def fill_random(self, generator: torch.Generator | None = None) -> None:
        """Generate random data matching dtype and resolved shape."""
        if self._resolved_shape is None:
            raise RuntimeError("shape not resolved")

        if self.dtype.is_floating_point:
            self.data = torch.randn(
                self._resolved_shape, dtype=self.dtype, generator=generator
            )
        else:
            # Integer types: use randint
            if self.dtype == torch.int8:
                lo, hi = -128, 127
            elif self.dtype == torch.uint8:
                lo, hi = 0, 255
            elif self.dtype == torch.int16:
                lo, hi = -32768, 32767
            elif self.dtype == torch.int32:
                lo, hi = -(2**30), 2**30
            elif self.dtype == torch.int64:
                lo, hi = -(2**30), 2**30
            else:
                lo, hi = 0, 255
            self.data = torch.randint(
                lo,
                hi + 1,
                self._resolved_shape,
                dtype=self.dtype,
                generator=generator,
            )

    def to_float(self) -> torch.Tensor:
        """Convert data to float32."""
        if self.data is None:
            raise RuntimeError("no data")
        return self.data.to(torch.float32)

    def set_address(self, addr: int) -> None:
        """Store physical address (Stage 4)."""
        self._address = addr

    def numel(self) -> int:
        """Return element count (requires resolved shape)."""
        if self._resolved_shape is None:
            raise RuntimeError("shape not resolved")
        return self._element_count

    # ── Golden data backward compat ──

    @property
    def _golden_data(self) -> torch.Tensor | None:
        """Backward-compat alias for .golden."""
        return self.golden

    @_golden_data.setter
    def _golden_data(self, value: torch.Tensor | None) -> None:
        self.golden = value

    # ── Verification ──

    def verify(self, golden: torch.Tensor | None = None) -> None:
        """Compare HW output against golden. Sets .verified and .max_diff.

        Raises VerificationError on mismatch.
        """
        from vten.errors import VerificationError

        if golden is not None:
            self.golden = golden
        if self.golden is None:
            raise VerificationError(
                f"no golden for tensor '{self.name}'", tensor=self.name
            )
        hw = self.cpu() if self.on_device else self.data
        if hw is None:
            raise VerificationError(
                f"no data for tensor '{self.name}'", tensor=self.name
            )

        hw_flat = hw.flatten().float()
        golden_flat = self.golden.flatten().float()
        self.max_diff = (hw_flat - golden_flat).abs().max().item()
        self.verified = True

        from vten.runtime.verifier import check_match

        check_match(self.name, hw, self.golden)

    # ── Device state (inference) ──

    @property
    def on_device(self) -> bool:
        """True if tensor data resides on FPGA device memory."""
        return self._bo is not None

    def cpu(self) -> torch.Tensor:
        """Transfer data from device to host and return torch.Tensor.

        If deserialized data is already cached on host (from STORE readback),
        returns it directly — avoids re-reading from BO which can return stale
        data in hw_emu. Falls back to BO sync+read if no cached data.
        """
        # Prefer pre-deserialized data (set by _wrap_outputs from STORE readback)
        if self.data is not None:
            return self.data
        if self._bo is None:
            raise RuntimeError("no data on host or device")
        # Sync FROM_DEVICE — import xrt constants lazily
        self._bo.sync(2)  # XCL_BO_SYNC_BO_FROM_DEVICE = 2
        raw = bytes(self._bo.read(self._bo_size))
        if self._deserialize_fn is not None:
            return self._deserialize_fn(raw)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    def numpy(self):
        """Transfer to host and return as numpy array."""
        return self.cpu().numpy()

    def _bind_bo(
        self,
        bo: Any,
        size: int,
        deserialize_fn: Callable[[bytes], torch.Tensor] | None = None,
    ) -> None:
        """Bind an XRT buffer object (called by InferenceSession)."""
        self._bo = bo
        self._bo_size = size
        self._deserialize_fn = deserialize_fn

    def _unbind_bo(self) -> None:
        """Release device buffer binding."""
        self._bo = None
        self._bo_size = 0
        self._deserialize_fn = None

    def describe(self) -> dict:
        """Human-readable tensor state for debugging."""
        info: dict[str, Any] = {
            "name": self.name,
            "shape": self._resolved_shape,
            "dtype": str(self.dtype),
            "has_data": self.data is not None,
            "on_device": self.on_device,
        }
        if self.data is not None:
            info["data_range"] = (float(self.data.min()), float(self.data.max()))
        return info

    def __repr__(self) -> str:
        parts = (
            f"Tensor(name={self.name!r}, shape={self.shape}, "
            f"dtype={self.dtype}, interface={self.interface!r}"
        )
        if self.direction is not None:
            parts += f", direction={self.direction!r}"
        return parts + ")"
