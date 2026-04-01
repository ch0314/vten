"""Kernel base class and RegisterHandle.

Spec reference: 00_data_models.md §3-4, 01_kernel_and_dsl.md §2

v2: forward(**inputs)->dict, compute_derived_params(self), default_params,
    __set_name__/__getattr__ for composite sub-kernel support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vten.kernel.tensor import Tensor

if TYPE_CHECKING:
    pass


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
    default_params: dict = {}

    _tensor_descriptors: dict[str, Tensor] = {}
    _register_handles: dict[str, RegisterHandle] = {}

    # Set by __set_name__ when used as sub-kernel in CompositeKernel body
    _sub_ref_name: str = ""

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

    def __set_name__(self, owner: type, name: str) -> None:
        """Descriptor protocol: remember attr name for composite sub-kernel ref."""
        self._sub_ref_name = name

    def __getattribute__(self, name: str) -> object:
        """Intercept tensor access on sub-kernel instances in CompositeKernel body.

        When this Kernel instance is used as `wl = WeightLoaderKernel()` in a
        CompositeKernel class body, `wl.wgt_out` returns a TensorRef for the
        >> connection operator.

        Only activates when _sub_ref_name is set (by __set_name__ or frame
        introspection during class body definition).
        """
        # Fast path: private attrs and known fields use normal lookup
        if name.startswith("_") or name in (
            "spec", "default_params", "connections",
        ):
            return object.__getattribute__(self, name)

        # Check if this is a tensor descriptor access on a sub-kernel ref
        cls = type(self)
        descriptors = cls.__dict__.get("_tensor_descriptors")
        if descriptors is None:
            for base in cls.__mro__:
                descriptors = base.__dict__.get("_tensor_descriptors")
                if descriptors is not None:
                    break
        if descriptors and name in descriptors:
            # Get sub_ref_name — only return TensorRef if already set
            try:
                sub_ref = object.__getattribute__(self, "_sub_ref_name")
            except AttributeError:
                sub_ref = ""
            if not sub_ref:
                # Try to discover from caller's frame, but ONLY inside a class
                # body that has __qualname__ (indicating class definition)
                import sys
                frame = sys._getframe(1)
                is_class_body = "__qualname__" in frame.f_locals
                if is_class_body:
                    for var_name, var_val in frame.f_locals.items():
                        if var_val is self and not var_name.startswith("_"):
                            sub_ref = var_name
                            object.__setattr__(self, "_sub_ref_name", var_name)
                            break
                del frame
            if sub_ref:
                from vten.kernel.composite import TensorRef
                return TensorRef(sub_ref, name, cls)

        return object.__getattribute__(self, name)

    def tensors(self) -> list[Tensor]:
        """Return list of all Tensor descriptors for this kernel.

        Returns instance-level copies if they exist (e.g., after
        copy.copy in KernelInstance.initialize), falling back to
        class-level descriptors.
        """
        result = []
        for name in self.__class__._tensor_descriptors:
            try:
                result.append(object.__getattribute__(self, name))
            except AttributeError:
                result.append(self.__class__._tensor_descriptors[name])
        return result

    def get_tensor(self, name: str) -> Tensor:
        """Get a tensor by name."""
        if name in self.__class__._tensor_descriptors:
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return self.__class__._tensor_descriptors[name]
        raise AttributeError(f"No tensor '{name}' in {self.__class__.__name__}")

    def compute_derived_params(self) -> dict:
        """Compute derived parameters from resolved base params.

        Override this to add computed values (e.g., shape calculations
        involving conditionals) to the parameter namespace before tensor
        shape resolution. Called during KernelInstance.initialize()
        after base params are set as instance attributes.

        Access params via self.param_name (e.g., self.out_ch, self.in_depth).

        Returns:
            Dict of additional params to merge into namespace.
        """
        return {}

    def generate_inputs(self, seed: int | None = None) -> None:
        """User-overrideable: generate input tensors."""
        raise NotImplementedError

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute golden reference outputs from inputs.

        Args:
            **inputs: {input_tensor_name: logical_data} for each input tensor.
        Returns:
            {output_tensor_name: logical_data} for each output tensor.
        """
        raise NotImplementedError

    def run(self, ctx: object) -> None:
        """User-overrideable: DUT-specific execution protocol.

        Override this to define the DSL sequence (send, recv, configure,
        write_register, verify) for this kernel. Called by @test_kernel
        or directly from test scenarios.

        Args:
            ctx: ExecutionContext instance.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement run(). "
            f"Override run(self, ctx) to define the execution protocol."
        )
