"""Tensor descriptor class.

Spec reference: 00_data_models.md §2

v2: shape/dtype are logical (algorithmic). Layout is handled by
kernel methods layout_{name}() / unlayout_{name}().
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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

        # instantiate() time (eager resolution) — logical shape
        self._resolved_shape: tuple[int, ...] | None = None
        self._element_count: int = 0

        # Logical data (algorithmic level)
        self._logical_data: torch.Tensor | None = None

        # compile() time (stages 1-4) — physical data
        self.data: torch.Tensor | None = None
        self._address: int | None = None

    # ── Logical data ──

    @property
    def logical_data(self) -> torch.Tensor | None:
        """Algorithmic-level data. Primary user interface."""
        return self._logical_data

    @logical_data.setter
    def logical_data(self, value: torch.Tensor) -> None:
        self._logical_data = value

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

    def describe(self) -> dict:
        """Human-readable tensor state for debugging."""
        info: dict[str, Any] = {
            "name": self.name,
            "shape": self._resolved_shape,
            "dtype": str(self.dtype),
            "has_data": self.data is not None,
        }
        if self._logical_data is not None:
            info["has_logical_data"] = True
            info["logical_data_range"] = (
                float(self._logical_data.min()),
                float(self._logical_data.max()),
            )
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
