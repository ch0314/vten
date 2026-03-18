"""Kernel base class and RegisterHandle.

Spec reference: 00_data_models.md §3-4, 01_kernel_and_dsl.md §2
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vten.kernel.tensor import Tensor


@dataclass
class RegisterHandle:
    """Lightweight handle for register interface reference."""

    interface_name: str


def register(interface_name: str) -> RegisterHandle:
    """Helper function for Kernel class declaration."""
    return RegisterHandle(interface_name)


class Kernel:
    """Single RTL module verification unit."""

    spec: str = ""

    _tensor_descriptors: dict[str, Tensor] = {}
    _register_handles: dict[str, RegisterHandle] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register Tensor & RegisterHandle descriptors."""
        super().__init_subclass__(**kwargs)
        cls._tensor_descriptors = {}
        cls._register_handles = {}
        for attr_name, attr_value in vars(cls).items():
            if isinstance(attr_value, Tensor):
                attr_value.name = attr_name
                cls._tensor_descriptors[attr_name] = attr_value
            elif isinstance(attr_value, RegisterHandle):
                cls._register_handles[attr_name] = attr_value

    def tensors(self) -> list[Tensor]:
        """Return list of all Tensor descriptors for this kernel."""
        return list(self.__class__._tensor_descriptors.values())

    def get_tensor(self, name: str) -> Tensor:
        """Get a tensor by name."""
        if name in self.__class__._tensor_descriptors:
            return getattr(self, name)
        raise AttributeError(f"No tensor '{name}' in {self.__class__.__name__}")

    def generate_inputs(self, seed: int | None = None) -> None:
        """User-overrideable: generate input tensors."""
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """User-overrideable: compute golden reference output."""
        raise NotImplementedError

    def verify(
        self, hw_output: torch.Tensor, golden: torch.Tensor
    ) -> bool:
        """Default verify: torch.allclose comparison."""
        return bool(torch.allclose(hw_output, golden))

    @classmethod
    def bind(
        cls,
        interface_map: dict,
        params: dict | None = None,
    ) -> SubKernelBinding:
        """Bind this Kernel as sub-kernel to CompositeKernel."""
        from vten.kernel.composite import SubKernelBinding

        return SubKernelBinding(
            kernel_class=cls,
            interface_map=interface_map,
            params=params,
        )


# Avoid circular import at module level
from vten.kernel.composite import SubKernelBinding  # noqa: E402, F401
