"""CompositeKernel, SubKernelBinding, Internal, TensorProxy, ExposedTensorDef, Connect.

Spec reference: 00_data_models.md §4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class Internal:
    """Marker for internal (RTL wire, no BFM) interface mapping."""

    def __init__(self, probe: bool = False) -> None:
        self.probe = probe


class TensorProxy:
    """Proxy for sub-kernel tensor access (class definition time).

    Stores a reference to the SubKernelBinding so that binding_attr_name
    resolves lazily (after __init_subclass__ sets _attr_name).
    """

    def __init__(
        self,
        binding: SubKernelBinding | str,
        tensor_name: str,
        kernel_class: type,
    ) -> None:
        self._binding = binding
        self.tensor_name = tensor_name
        self.kernel_class = kernel_class

    @property
    def binding_attr_name(self) -> str:
        if isinstance(self._binding, str):
            return self._binding
        return self._binding._attr_name

    def expose(self, interface: str) -> ExposedTensorDef:
        """Expose sub-kernel tensor to CompositeKernel top level."""
        return ExposedTensorDef(
            proxy=self,
            top_interface=interface,
        )


class ExposedTensorDef:
    """Result of expose() call. Collected by __init_subclass__."""

    def __init__(self, proxy: TensorProxy, top_interface: str) -> None:
        self._proxy = proxy
        self.top_interface = top_interface

    @property
    def origin_sub_kernel(self) -> str:
        return self._proxy.binding_attr_name

    @property
    def origin_name(self) -> str:
        return self._proxy.tensor_name


@dataclass
class SubKernelBinding:
    """Result of Kernel.bind(). Stored as CompositeKernel attribute."""

    kernel_class: type
    interface_map: dict
    params: dict | None = None
    _attr_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = name

    def __getattr__(self, name: str) -> TensorProxy:
        """Access sub-kernel tensor by proxy (class definition time)."""
        if name.startswith("_") or name in (
            "kernel_class",
            "interface_map",
            "params",
        ):
            raise AttributeError(name)

        from vten.kernel.tensor import Tensor

        attr = getattr(self.kernel_class, name, None)
        if isinstance(attr, Tensor):
            return TensorProxy(
                binding=self,
                tensor_name=name,
                kernel_class=self.kernel_class,
            )
        raise AttributeError(
            f"no tensor '{name}' in {self.kernel_class.__name__}"
        )


class Connect:
    """RTL internal wire descriptor for CompositeKernel.connections."""

    def __init__(
        self,
        source: TensorProxy,
        dest: TensorProxy,
        transform: Any | None = None,
    ) -> None:
        if not isinstance(source, TensorProxy):
            raise TypeError(
                f"Connect source must be TensorProxy, got {type(source)}"
            )
        if not isinstance(dest, TensorProxy):
            raise TypeError(
                f"Connect dest must be TensorProxy, got {type(dest)}"
            )

        self._source_proxy = source
        self._dest_proxy = dest
        self.source_name = source.tensor_name
        self.dest_name = dest.tensor_name
        self.transform = transform
        self.is_internal_wire = True  # RTL wire — skip dtype/shape validation

        from vten.kernel.tensor import Tensor

        source_tensor = getattr(source.kernel_class, source.tensor_name, None)
        self.source_interface = (
            source_tensor.interface if isinstance(source_tensor, Tensor) else None
        )

        dest_tensor = getattr(dest.kernel_class, dest.tensor_name, None)
        self.dest_interface = (
            dest_tensor.interface if isinstance(dest_tensor, Tensor) else None
        )

    @property
    def source_sub(self) -> str:
        return self._source_proxy.binding_attr_name

    @property
    def dest_sub(self) -> str:
        return self._dest_proxy.binding_attr_name


def _topo_sort(connections: list[Connect], bindings: dict[str, SubKernelBinding]) -> list[str]:
    """Topological sort of sub-kernels based on connection graph.

    Returns sub-kernel binding names in dependency order (sources before sinks).
    Sub-kernels with no connections appear at the end.
    """
    # Build adjacency: source_sub → dest_sub
    all_names = set(bindings.keys())
    in_edges: dict[str, set[str]] = {n: set() for n in all_names}
    out_edges: dict[str, set[str]] = {n: set() for n in all_names}

    for conn in connections:
        src, dst = conn.source_sub, conn.dest_sub
        if src in all_names and dst in all_names and src != dst:
            out_edges[src].add(dst)
            in_edges[dst].add(src)

    # Kahn's algorithm
    queue = [n for n in all_names if not in_edges[n]]
    queue.sort()  # deterministic order for equal-priority nodes
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in sorted(out_edges[node]):
            in_edges[neighbor].discard(node)
            if not in_edges[neighbor]:
                queue.append(neighbor)

    # Append any remaining (cycle or disconnected) — shouldn't happen in practice
    for n in sorted(all_names - set(result)):
        result.append(n)

    return result


def _make_composite_kernel():
    """Deferred import to avoid circular dependency with Kernel."""
    from vten.kernel.base import Kernel

    class CompositeKernel(Kernel):
        """Multi-sub-kernel higher-level verification unit."""

        connections: list[Connect] = []

        _sub_kernel_bindings: dict[str, SubKernelBinding] = {}
        _exposed_tensor_defs: dict[str, ExposedTensorDef] = {}
        _connections: list[Connect] = []

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            cls._sub_kernel_bindings = {}
            cls._exposed_tensor_defs = {}
            cls._connections = []

            for attr_name, attr_value in vars(cls).items():
                if isinstance(attr_value, SubKernelBinding):
                    attr_value._attr_name = attr_name
                    cls._sub_kernel_bindings[attr_name] = attr_value
                elif isinstance(attr_value, ExposedTensorDef):
                    cls._exposed_tensor_defs[attr_name] = attr_value
                elif attr_name == "connections" and isinstance(attr_value, list):
                    cls._connections = attr_value

        def bindings(self) -> list[tuple[str, SubKernelBinding]]:
            return list(self.__class__._sub_kernel_bindings.items())

        def exposed_tensor_defs(self) -> list[tuple[str, ExposedTensorDef]]:
            return list(self.__class__._exposed_tensor_defs.items())

        @classmethod
        def topo_sort_sub_kernels(cls) -> list[str]:
            """Return sub-kernel names in topological (dependency) order."""
            return _topo_sort(cls._connections, cls._sub_kernel_bindings)

    return CompositeKernel


# Lazy singleton
_CompositeKernel = None


def __getattr__(name: str):
    """Module-level lazy import for CompositeKernel."""
    global _CompositeKernel
    if name == "CompositeKernel":
        if _CompositeKernel is None:
            _CompositeKernel = _make_composite_kernel()
        return _CompositeKernel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
