"""CompositeKernel, TensorRef, Connection.

Spec reference: 00_data_models.md §4, 10_kernel_v2_design.md §5

v2: Sub-kernels declared as Kernel(), connections via >> operator,
    auto-expose, auto-forward chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import torch


# ── TensorRef & Connection ──


class TensorRef:
    """Reference to a sub-kernel tensor, created by Kernel.__getattr__.

    Used in CompositeKernel class body for connection declarations:
        connections = [wl.wgt_out >> mac.weight]
    """

    def __init__(self, sub_name: str, tensor_name: str, kernel_class: type) -> None:
        self.sub_name = sub_name
        self.tensor_name = tensor_name
        self.kernel_class = kernel_class

    def __rshift__(self, other: TensorRef) -> Connection:
        if not isinstance(other, TensorRef):
            raise TypeError(
                f"Connection dest must be TensorRef, got {type(other)}"
            )
        return Connection(source=self, dest=other)

    def __repr__(self) -> str:
        return f"TensorRef({self.sub_name}.{self.tensor_name})"


@dataclass
class Connection:
    """Internal wire between two sub-kernel tensors."""

    source: TensorRef
    dest: TensorRef

    @property
    def source_sub(self) -> str:
        return self.source.sub_name

    @property
    def dest_sub(self) -> str:
        return self.dest.sub_name

    @property
    def source_name(self) -> str:
        return self.source.tensor_name

    @property
    def dest_name(self) -> str:
        return self.dest.tensor_name

    @property
    def source_interface(self) -> str | None:
        from vten.kernel.tensor import Tensor

        t = getattr(self.source.kernel_class, self.source.tensor_name, None)
        return t.interface if isinstance(t, Tensor) else None

    @property
    def dest_interface(self) -> str | None:
        from vten.kernel.tensor import Tensor

        t = getattr(self.dest.kernel_class, self.dest.tensor_name, None)
        return t.interface if isinstance(t, Tensor) else None

    # Backward compat with code checking these attributes
    is_internal_wire: bool = True
    transform: object = None


# ── Topological sort ──


def _topo_sort(connections: list[Connection], sub_names: set[str]) -> list[str]:
    """Topological sort of sub-kernels based on connection graph.

    Returns sub-kernel names in dependency order (sources before sinks).
    Sub-kernels with no connections appear at the end.
    """
    in_edges: dict[str, set[str]] = {n: set() for n in sub_names}
    out_edges: dict[str, set[str]] = {n: set() for n in sub_names}

    for conn in connections:
        src, dst = conn.source_sub, conn.dest_sub
        if src in sub_names and dst in sub_names and src != dst:
            out_edges[src].add(dst)
            in_edges[dst].add(src)

    # Kahn's algorithm
    queue = sorted(n for n in sub_names if not in_edges[n])
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in sorted(out_edges[node]):
            in_edges[neighbor].discard(node)
            if not in_edges[neighbor]:
                queue.append(neighbor)

    # Append remaining (cycle or disconnected)
    for n in sorted(sub_names - set(result)):
        result.append(n)

    return result


# ── CompositeKernel ──


def _make_composite_kernel():
    """Deferred import to avoid circular dependency with Kernel."""
    from vten.kernel.base import Kernel
    from vten.kernel.tensor import Tensor

    class CompositeKernel(Kernel):
        """Multi-sub-kernel higher-level verification unit.

        v2 API:
            wl   = WeightLoaderKernel()
            mac  = MacAtuKernel()
            connections = [wl.wgt_out >> mac.weight]
            # Tensors not in connections → auto-exposed
        """

        connections: list[Connection] = []

        _sub_kernel_refs: dict[str, type] = {}  # attr_name → kernel_class
        _connections: list[Connection] = []
        _connected_tensors: set[tuple[str, str]] = set()  # (sub_name, tensor_name)
        _auto_exposed: dict[tuple[str, str], str] = {}  # (sub, tensor) → tensor_name

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            cls._sub_kernel_refs = {}
            cls._connections = []
            cls._connected_tensors = set()
            cls._auto_exposed = {}

            # Collect sub-kernel refs: Kernel instances in class body
            for attr_name in list(vars(cls)):
                attr_value = vars(cls)[attr_name]
                if (
                    isinstance(attr_value, Kernel)
                    and not isinstance(attr_value, CompositeKernel)
                    and attr_name != "connections"
                ):
                    cls._sub_kernel_refs[attr_name] = type(attr_value)

            # Collect connections
            conns = vars(cls).get("connections")
            if isinstance(conns, list):
                cls._connections = conns

            # Compute connected tensor set
            for conn in cls._connections:
                cls._connected_tensors.add((conn.source_sub, conn.source_name))
                cls._connected_tensors.add((conn.dest_sub, conn.dest_name))

            # Auto-expose: tensors NOT in connections
            for ref_name, ref_class in cls._sub_kernel_refs.items():
                for t_name in ref_class._tensor_descriptors:
                    if (ref_name, t_name) not in cls._connected_tensors:
                        cls._auto_exposed[(ref_name, t_name)] = t_name

        @classmethod
        def topo_sort_sub_kernels(cls) -> list[str]:
            """Return sub-kernel names in topological (dependency) order."""
            return _topo_sort(cls._connections, set(cls._sub_kernel_refs.keys()))

        def compute_derived_params(self) -> dict:
            """Auto-chain: call each sub-kernel's compute_derived_params and merge."""
            derived: dict = {}
            for _attr_name, sub_cls in self.__class__._sub_kernel_refs.items():
                # Create temporary instance to call compute_derived_params
                tmp = sub_cls()
                # Copy params from self to tmp
                for k, v in vars(self).items():
                    if not k.startswith("_") and isinstance(v, (int, float, str)):
                        if not hasattr(tmp, k):
                            setattr(tmp, k, v)
                sub_derived = tmp.compute_derived_params()
                derived.update(sub_derived)
            return derived

        def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            """Auto-chain: connection graph + sub-kernel forward() calls.

            Multi-round dataflow evaluation handles cycles (e.g., fmapIO).

            Pipeline per sub-kernel:
              pool(logical) → layout_{input}() → physical
              → sub.forward(**physical) → physical outputs
              → unlayout_{output}() → logical → pool / connections

            Exposed outputs are returned as physical (forward() always
            returns physical data, matching DUT behavior).
            """
            cls = self.__class__
            MAX_ROUNDS = 20

            # pool stores LOGICAL data for connection propagation
            pool: dict[tuple[str, str], Any] = {}  # (sub_name, tensor_name) → logical
            # physical_pool stores PHYSICAL forward() outputs for exposed return
            physical_pool: dict[tuple[str, str], Any] = {}

            # 1. Seed exposed input tensors from kwargs (logical)
            for (sub_name, tensor_name), _t_name in cls._auto_exposed.items():
                exposed_key = f"{sub_name}_{tensor_name}"
                if tensor_name in inputs:
                    pool[(sub_name, tensor_name)] = inputs[tensor_name]
                elif exposed_key in inputs:
                    pool[(sub_name, tensor_name)] = inputs[exposed_key]

            # Also check instance attributes for logical_data
            for (sub_name, tensor_name) in cls._auto_exposed:
                if (sub_name, tensor_name) in pool:
                    continue
                sub_inst = self._get_sub_kernel_instance(sub_name)
                if sub_inst is not None:
                    t = sub_inst.get_tensor(tensor_name)
                    if t.logical_data is not None:
                        pool[(sub_name, tensor_name)] = t.logical_data

            # 2. Multi-round dataflow evaluation
            sub_order = cls.topo_sort_sub_kernels()
            computed_outputs: set[tuple[str, str]] = set()

            for _round in range(MAX_ROUNDS):
                progress = False
                for sub_name in sub_order:
                    sub_inst = self._get_sub_kernel_instance(sub_name)
                    if sub_inst is None:
                        continue

                    # Collect available logical inputs for this sub-kernel
                    available: dict[str, Any] = {}
                    for t in sub_inst.tensors():
                        key = (sub_name, t.name)
                        if key in pool and key not in computed_outputs:
                            available[t.name] = pool[key]

                    if not available:
                        continue

                    # Defer if connected inputs are missing but their
                    # sources could still produce them in a later round.
                    # Exception: cycle-breaking kernels that also have
                    # exposed (non-connected) inputs can run with partial
                    # input — they implement ``if "x" in inputs`` guards.
                    has_missing_connected = False
                    for conn in cls._connections:
                        if conn.dest_sub == sub_name:
                            if (conn.dest_sub, conn.dest_name) not in pool:
                                has_missing_connected = True
                                break
                    if has_missing_connected:
                        has_exposed_input = any(
                            t.name in available
                            for t in sub_inst.tensors()
                            if (sub_name, t.name) not in cls._connected_tensors
                        )
                        if not has_exposed_input:
                            continue

                    # Layout exposed inputs: logical → physical only for
                    # tensors that enter from outside (auto-exposed).
                    # Connected inputs stay logical — they came from
                    # another sub-kernel's forward() output.
                    forward_inputs: dict[str, Any] = {}
                    for name, data in available.items():
                        key = (sub_name, name)
                        if key not in cls._connected_tensors:
                            # Exposed input — apply layout if defined
                            layout_fn = getattr(
                                sub_inst, f"layout_{name}", None,
                            )
                            forward_inputs[name] = (
                                layout_fn(data)
                                if layout_fn is not None
                                else data
                            )
                        else:
                            # Connected input — pass logical as-is
                            forward_inputs[name] = data
                    try:
                        outputs = sub_inst.forward(**forward_inputs)
                    except (NotImplementedError, TypeError):
                        continue

                    for out_name, out_data in outputs.items():
                        out_key = (sub_name, out_name)
                        if out_key not in computed_outputs:
                            # For exposed outputs: apply unlayout to get
                            # logical; store physical in physical_pool.
                            # For connected outputs: data is already
                            # logical; no unlayout needed.
                            if out_key in cls._connected_tensors:
                                pool[out_key] = out_data
                            else:
                                physical_pool[out_key] = out_data
                                unlayout_fn = getattr(
                                    sub_inst, f"unlayout_{out_name}", None,
                                )
                                pool[out_key] = (
                                    unlayout_fn(out_data)
                                    if unlayout_fn is not None
                                    else out_data
                                )
                            computed_outputs.add(out_key)
                            progress = True
                            # Propagate through connections (logical)
                            for conn in cls._connections:
                                if (
                                    conn.source_sub == sub_name
                                    and conn.source_name == out_name
                                ):
                                    pool[
                                        (conn.dest_sub, conn.dest_name)
                                    ] = pool[out_key]

                if not progress:
                    break

            # 3. Collect exposed outputs (physical — forward always returns physical)
            result: dict[str, torch.Tensor] = {}
            for (sub_name, tensor_name) in cls._auto_exposed:
                key = (sub_name, tensor_name)
                if key in computed_outputs:
                    result[tensor_name] = physical_pool[key]

            return result

        def _get_sub_kernel_instance(self, sub_name: str) -> Kernel | None:
            """Get the actual kernel instance for a sub-kernel."""
            # After KernelInstance initialization, sub-kernel instances are stored
            # via _kernel_instance._sub_kernel_instances
            ki = getattr(self, "_kernel_instance", None)
            if ki is not None:
                sub_kis = getattr(ki, "_sub_kernel_instances", None)
                if sub_kis and sub_name in sub_kis:
                    return sub_kis[sub_name].kernel_class_instance
            return None

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
