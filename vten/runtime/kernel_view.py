"""Stage 0: Composite Kernel Flattening.

Data structures: InterfaceMapping, ExposedTensor, ProbePoint,
KernelInstance, FlattenedKernelView.

Spec reference: 00_data_models.md §7, 02_runtime_engine.md §5

v2: SubKernelBinding → _sub_kernel_refs, auto-expose,
    KernelInstance.initialize() order change for compute_derived_params(self).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

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
    from vten.kernel.composite import Connection
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
    """Tensor exposed from CompositeKernel (auto-exposed or manually)."""

    name: str
    origin_path: str  # "dma_ifm.src"
    origin_tensor: Tensor
    top_interface: str
    direction: Direction

    # Mutable state set during compilation
    _serialized: bytes | None = None
    _serialized_size: int = 0
    _port_buffers: dict[str, bytes] | None = None  # port_name → data chunk
    _port_mode: str = "block"  # "block" | "channel_interleave"
    _interleave_unit: int | None = None

    @property
    def data(self) -> torch.Tensor | None:
        return self.origin_tensor.data

    @data.setter
    def data(self, value: torch.Tensor | None) -> None:
        self.origin_tensor.data = value

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.origin_tensor._resolved_shape

    @property
    def element_count(self) -> int | None:
        return self.origin_tensor._element_count

    @property
    def address(self) -> int | None:
        return self.origin_tensor._address

    def set_address(self, addr: int) -> None:
        self.origin_tensor._address = addr

    def fill_random(self, generator: torch.Generator | None = None) -> None:
        self.origin_tensor.fill_random(generator=generator)

    @property
    def dtype(self) -> torch.dtype:
        return self.origin_tensor.dtype



# ── ProbePoint ──


@dataclass
class ProbePoint:
    """Golden data for probe verification.

    Two usage modes:
    - Composite internal probe: connection + interface_mapping set.
    - Single kernel probe (probe=True on PULL): tensor_name set.
    """

    connection: Connection | None = None
    interface_mapping: InterfaceMapping | None = None
    tensor_name: str | None = None
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

    def initialize(self, project_params: dict, *, project_dir: Path | None = None) -> None:
        """Initialize: resolve parameters + shapes, create Kernel instance.

        v2 order:
        1. Create ParameterResolver (with default_params from Kernel class)
        2. Create kernel instance
        3. Set base param attrs on instance
        4. Call instance.compute_derived_params() → merge derived
        5. Set derived attrs on instance
        6. Copy tensors and resolve shapes
        7. Resolve exposed tensors for CompositeKernel
        """
        from vten.runtime.resolver import ParameterResolver

        # Merge build_params: project [build_params] < kernel_spec build_params
        project_build = project_params.get("build_params", {}) or {}
        spec_build = self.spec.build_params or {}
        merged_build = {**project_build, **spec_build} if (project_build or spec_build) else None

        # Get default_params from kernel class
        default_params = getattr(self.kernel_class, "default_params", None) or None

        self._resolver = ParameterResolver(
            project_params,
            self.spec.parameters,
            self.runtime_params,
            build_params=merged_build,
            default_params=default_params,
        )

        # Create instance first (v2: before compute_derived_params)
        self.kernel_class_instance = self.kernel_class()
        self.kernel_class_instance._kernel_instance = self

        # Set base param attrs on instance
        for key, value in self._resolver.namespace.items():
            if not hasattr(self.kernel_class_instance, key) or not key.startswith("_"):
                setattr(self.kernel_class_instance, key, value)

        # Compute derived params (v2: instance method, self.* access)
        derived = self.kernel_class_instance.compute_derived_params()
        if derived:
            self._resolver.namespace.update(derived)
            # Set derived attrs on instance
            for key, value in derived.items():
                setattr(self.kernel_class_instance, key, value)

        # Copy tensors and resolve shapes (logical)
        for tensor in self.kernel_class_instance.tensors():
            instance_tensor = copy.copy(tensor)
            setattr(self.kernel_class_instance, tensor.name, instance_tensor)
            instance_tensor._resolve_shape(self._resolver)

        # Resolve ExposedTensor for CompositeKernel
        self._resolve_exposed_tensors(project_params, project_dir=project_dir)

    def _resolve_exposed_tensors(
        self, project_params: dict, *, project_dir: Path | None = None,
    ) -> None:
        """For CompositeKernel: create sub-kernel instances for auto-exposed tensors."""
        inst = self.kernel_class_instance
        sub_kernel_refs = getattr(inst.__class__, "_sub_kernel_refs", {})
        if not sub_kernel_refs:
            return

        # Instantiate sub-kernels as proper KernelInstance objects
        self._sub_kernel_instances = {}
        for attr_name, sub_cls in sub_kernel_refs.items():
            merged_params = dict(self.runtime_params)
            sub_spec_path = getattr(sub_cls, "spec", "")
            sub_spec = None
            if sub_spec_path:
                try:
                    from vten.spec.parser import load_kernel_spec
                    if project_dir is not None:
                        spec_file = project_dir / sub_spec_path
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
            sub_ki.initialize(project_params, project_dir=project_dir)
            self._sub_kernel_instances[attr_name] = sub_ki

        # Create ExposedTensor proxies for auto-exposed tensors
        auto_exposed = getattr(inst.__class__, "_auto_exposed", {})
        for (sub_attr, tensor_name), _t_name in auto_exposed.items():
            sub_ki = self._sub_kernel_instances.get(sub_attr)
            if sub_ki is None:
                continue
            origin_tensor = sub_ki.get_tensor(tensor_name)
            direction = getattr(origin_tensor, "direction", None)
            if direction is None:
                direction = Direction.HOST_TO_DEV
            exposed = ExposedTensor(
                name=tensor_name,
                origin_path=f"{sub_attr}.{tensor_name}",
                origin_tensor=origin_tensor,
                top_interface=origin_tensor.interface,
                direction=direction,
            )
            setattr(inst, f"_exposed_{sub_attr}_{tensor_name}", exposed)
            # Also set short name so user code can access self.data_in directly
            if not hasattr(inst, tensor_name) or isinstance(
                getattr(type(inst), tensor_name, None), Tensor
            ):
                setattr(inst, tensor_name, exposed)

        # Also set sub-kernel instances accessible via composite
        for attr_name, sub_ki in self._sub_kernel_instances.items():
            # Set sub-kernel instance so composite can access sub.tensor
            setattr(inst, attr_name, sub_ki.kernel_class_instance)

    def tensors(self) -> list[Tensor]:
        """Return all Tensor descriptors from the underlying kernel instance."""
        if self.kernel_class_instance:
            return self.kernel_class_instance.tensors()
        return []

    def get_tensor(self, name: str) -> Tensor:
        """Look up a tensor by name. Raises RuntimeError if not initialized."""
        if self.kernel_class_instance:
            return self.kernel_class_instance.get_tensor(name)
        raise RuntimeError(f"KernelInstance '{self.name}' not initialized")

    def __getattr__(self, name: str) -> object:
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
    connections: list[Connection]

    _top_resolver: ParameterResolver | None = None
    _register_bindings: list | None = None  # list[RegisterBindingEntry]

    def external_interfaces(self) -> list[str]:
        """Top-level interface names that need BFM generation."""
        seen: set[str] = set()
        result: list[str] = []
        for m in self.interface_mappings:
            if m.mapping_type == MappingType.EXTERNAL:
                if m.top_interface not in seen:
                    seen.add(m.top_interface)
                    result.append(m.top_interface)
        return result

    def tensors_for_interface(self, top_iface: str) -> list[ExposedTensor]:
        """All exposed tensors bound to a top-level interface."""
        return [
            t for t in self.exposed_tensors.values() if t.top_interface == top_iface
        ]

    def quant_for_tensor(self, name: str):
        """Declared QuantSpec for an exposed tensor's interface, or None.

        Resolves through the origin sub-kernel's spec (``"_self"`` for unit
        kernels), so a composite output picks up the quant block declared on
        the *sub-kernel* interface that produced it.
        """
        exposed = self.exposed_tensors.get(name)
        if exposed is None:
            return None
        sub_name = exposed.origin_path.split(".", 1)[0]
        sub_ki = self.sub_kernels.get(sub_name)
        spec = sub_ki.spec if sub_ki is not None else self.top_spec
        iface = spec.interfaces.get(exposed.origin_tensor.interface)
        return iface.quant if iface is not None else None

    def registers_for_interface(
        self, top_iface: str
    ) -> list[tuple[str, RegisterSpec, int]]:
        """All registers for a top-level interface.
        Returns: [(sub_kernel_name, register_spec, absolute_offset)]"""
        result: list[tuple[str, RegisterSpec, int]] = []
        for m in self.interface_mappings:
            if m.top_interface != top_iface:
                continue
            if m.mapping_type != MappingType.EXTERNAL:
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
            f"'{sub_kernel_name}', but no matching exposed tensor found."
        )
