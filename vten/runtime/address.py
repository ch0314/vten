"""Stage 4: Address Allocation.

Spec reference: 02_runtime_engine.md §10
"""

from __future__ import annotations

from vten.errors import MemoryOverflowError
from vten.spec.models import MemoryRegion


class AddressAllocator:
    """Sequential address allocator for a memory region."""

    def __init__(self, region: MemoryRegion) -> None:
        self.region = region
        self.next_addr = region.base
        self._allocated_addrs: dict[str, int] = {}

    def allocate(self, tensor_name: str, size: int) -> int:
        """Allocate memory for a tensor. Returns aligned address."""
        aligned = self._align_up(self.next_addr, self.region.alignment)
        if aligned + size > self.region.base + self.region.size:
            raise MemoryOverflowError(
                f"{tensor_name} exceeds {self.region.name}"
            )
        self._allocated_addrs[tensor_name] = aligned
        self.next_addr = aligned + size
        return aligned

    @staticmethod
    def _align_up(addr: int, alignment: int) -> int:
        return (addr + alignment - 1) & ~(alignment - 1)
