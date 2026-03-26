"""Stage 0: Composite Kernel Flattening.

Data structures: InterfaceMapping, ExposedTensor, ProbePoint,
KernelInstance, FlattenedKernelView.

Spec reference: 00_data_models.md §7, 02_runtime_engine.md §5
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.errors import BindingError, ValidationError
from vten.kernel.tensor import Tensor
from vten.spec.models import (
    Direction,
    KernelSpec,
    MappingType,
    Protocol,
    RegisterSpec,
    Role,
)

if TYPE_CHECKING:
    from vten.kernel.composite import Connect
    from vten.runtime.resolver import ParameterResolver


# ── InterfaceMapping ──


@dataclass
class InterfaceMapping:
    """Sub-kernel interface → top-level mapping result."""

    sub_kernel: str
    sub_interface: str
    mapping_type: MappingType
    top_interface: str | None
    bank_name: str | None
    bank_offset: int = 0


# ── ExposedTensor ──


@dataclass
class ExposedTensor:
    """Tensor exposed from CompositeKernel via .expose()."""

    name: str
    origin_path: str  # "dma_ifm.src"
    origin_tensor: Tensor
    top_interface: str
    direction: Direction

    # Mutable state set during compilation
    _serialized: bytes | None = None
    _serialized_size: int = 0
    _split_buffers: dict[str, bytes] | None = None
    _array_element_buffers: dict[str, bytes] | None = None  # flat_name → data chunk

    @property
    def data(self):
        return self.origin_tensor.data

    @data.setter
    def data(self, value):
        self.origin_tensor.data = value

    @property
    def shape(self):
        return self.origin_tensor._resolved_shape

    @property
    def element_count(self):
        return self.origin_tensor._element_count

    @property
    def address(self):
        return self.origin_tensor._address

    def set_address(self, addr: int) -> None:
        self.origin_tensor._address = addr

    def fill_random(self, generator=None) -> None:
        self.origin_tensor.fill_random(generator=generator)

    @property
    def dtype(self):
        return self.origin_tensor.dtype


# ── ProbePoint ──


@dataclass
class ProbePoint:
    """Golden data container for Internal(probe=True) interfaces."""

    connection: Connect
    interface_mapping: InterfaceMapping
    golden_data: object = None  # torch.Tensor
    serialized_golden: bytes | None = None
    golden_buffer_id: int | None = None


# ── KernelInstance ──


@dataclass
class KernelInstance:
    """Kernel instance created by instantiate(). Eager resolution."""

    name: str
    spec: KernelSpec
    kernel_class: type
    kernel_class_instance: object | None = None
    runtime_params: dict = field(default_factory=dict)
    _resolver: ParameterResolver | None = None
    _sub_kernel_instances: dict[str, KernelInstance] | None = None

    def initialize(self, project_params: dict) -> None:
        """Initialize: resolve parameters + shapes, create Kernel instance."""
        from vten.runtime.resolver import ParameterResolver

        self._resolver = ParameterResolver(
            project_params,
            self.spec.parameters,
            self.runtime_params,
        )

        self.kernel_class_instance = self.kernel_class()

        for tensor in self.kernel_class_instance.tensors():
            instance_tensor = copy.copy(tensor)
            setattr(self.kernel_class_instance, tensor.name, instance_tensor)
            instance_tensor._resolve_shape(self._resolver)

        # Resolve ExposedTensorDef → ExposedTensor for CompositeKernel
        self._resolve_exposed_tensors(project_params)

        # Expose resolved params as instance attributes
        for key, value in self._resolver.namespace.items():
            if not hasattr(self.kernel_class_instance, key):
                setattr(self.kernel_class_instance, key, value)

    def _resolve_exposed_tensors(self, project_params: dict) -> None:
        """For CompositeKernel: resolve ExposedTensorDef to ExposedTensor.

        Creates proper KernelInstance objects for each sub-kernel and stores
        them in self._sub_kernel_instances so _flatten_composite() can reuse
        them instead of creating duplicates.
        """
        from vten.kernel.composite import ExposedTensorDef, SubKernelBinding

        inst = self.kernel_class_instance
        exposed_defs = getattr(inst.__class__, "_exposed_tensor_defs", {})
        if not exposed_defs:
            return

        # Instantiate sub-kernels as proper KernelInstance objects
        self._sub_kernel_instances = {}
        bindings = getattr(inst.__class__, "_sub_kernel_bindings", {})
        for attr_name, binding in bindings.items():
            sub_cls = binding.kernel_class
            sub_params = binding.params or {}
            merged_params = {**self.runtime_params, **sub_params}
            sub_spec_path = getattr(sub_cls, "spec", "")
            sub_spec = None
            if sub_spec_path:
                try:
                    from pathlib import Path as _Path
                    from vten.spec.parser import load_kernel_spec
                    project_dir = project_params.get("_project_dir")
                    if project_dir:
                        spec_file = _Path(project_dir) / sub_spec_path
                    else:
                        spec_file = sub_spec_path
                    sub_spec = load_kernel_spec(spec_file)
                except FileNotFoundError:
                    pass
            if sub_spec is None:
                sub_spec = KernelSpec(
                    kernel_name=sub_cls.__name__,
                    rtl_top=sub_cls.__name__,
                )
            sub_ki = KernelInstance(
                name=attr_name,
                spec=sub_spec,
                kernel_class=sub_cls,
                runtime_params=merged_params,
            )
            sub_ki.initialize(project_params)
            self._sub_kernel_instances[attr_name] = sub_ki

        # Replace ExposedTensorDef with ExposedTensor on the instance
        for attr_name, edef in exposed_defs.items():
            sub_attr = edef.origin_sub_kernel
            tensor_name = edef.origin_name
            sub_ki = self._sub_kernel_instances.get(sub_attr)
            if sub_ki is None:
                continue
            origin_tensor = sub_ki.get_tensor(tensor_name)
            # Infer direction from origin tensor
            direction = getattr(origin_tensor, "direction", None)
            if direction is None:
                direction = Direction.HOST_TO_DEV
            exposed = ExposedTensor(
                name=attr_name,
                origin_path=f"{sub_attr}.{tensor_name}",
                origin_tensor=origin_tensor,
                top_interface=edef.top_interface,
                direction=direction,
            )
            setattr(inst, attr_name, exposed)

    def tensors(self) -> list[Tensor]:
        if self.kernel_class_instance:
            return self.kernel_class_instance.tensors()
        return []

    def get_tensor(self, name: str) -> Tensor:
        if self.kernel_class_instance:
            return self.kernel_class_instance.get_tensor(name)
        raise RuntimeError(f"KernelInstance '{self.name}' not initialized")

    def __getattr__(self, name: str):
        """Delegate to kernel_class_instance for attribute access."""
        # Avoid infinite recursion for dataclass internals
        if name.startswith("_") or name in (
            "name",
            "spec",
            "kernel_class",
            "kernel_class_instance",
            "runtime_params",
        ):
            raise AttributeError(name)
        if self.kernel_class_instance is not None:
            return getattr(self.kernel_class_instance, name)
        raise AttributeError(
            f"KernelInstance '{self.name}' not initialized. "
            f"Attribute '{name}' not accessible."
        )


# ── FlattenedKernelView ──


@dataclass
class FlattenedKernelView:
    """Flattened representation used by all pipeline stages."""

    name: str
    top_spec: KernelSpec
    sub_kernels: dict[str, KernelInstance]
    interface_mappings: list[InterfaceMapping]
    exposed_tensors: dict[str, ExposedTensor]
    probe_points: list[ProbePoint]
    connections: list[Connect]

    _top_resolver: ParameterResolver | None = None
    _register_bindings: list | None = None  # list[RegisterBindingEntry]

    def external_interfaces(self) -> list[str]:
        """Top-level interface names that need BFM generation."""
        seen: set[str] = set()
        result: list[str] = []
        for m in self.interface_mappings:
            if m.mapping_type in (MappingType.EXTERNAL, MappingType.EXTERNAL_BANK):
                if m.top_interface not in seen:
                    seen.add(m.top_interface)
                    result.append(m.top_interface)
        return result

    def tensors_for_interface(self, top_iface: str) -> list[ExposedTensor]:
        """All exposed tensors bound to a top-level interface."""
        return [
            t for t in self.exposed_tensors.values() if t.top_interface == top_iface
        ]

    def registers_for_interface(
        self, top_iface: str
    ) -> list[tuple[str, RegisterSpec, int]]:
        """All registers for a top-level interface.
        Returns: [(sub_kernel_name, register_spec, absolute_offset)]"""
        result: list[tuple[str, RegisterSpec, int]] = []
        for m in self.interface_mappings:
            if m.top_interface != top_iface:
                continue
            if m.mapping_type not in (MappingType.EXTERNAL, MappingType.EXTERNAL_BANK):
                continue
            sub = self.sub_kernels[m.sub_kernel]
            for reg in sub.spec.get_registers(m.sub_interface):
                abs_offset = m.bank_offset + reg.offset
                result.append((m.sub_kernel, reg, abs_offset))
        return result

    def resolve_auto_bind_tensor(
        self, sub_kernel_name: str, tensor_name: str
    ) -> ExposedTensor:
        """Reverse lookup: sub-kernel tensor name → ExposedTensor."""
        origin_path = f"{sub_kernel_name}.{tensor_name}"
        for exposed in self.exposed_tensors.values():
            if exposed.origin_path == origin_path:
                return exposed
        raise BindingError(
            f"auto_bind references tensor '{tensor_name}' in sub-kernel "
            f"'{sub_kernel_name}', but no matching exposed tensor found. "
            f"Ensure the tensor is exposed via .expose() in the "
            f"CompositeKernel definition."
        )
