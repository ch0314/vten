"""CompositeKernel, TensorRef, Connection.

Spec reference: 00_data_models.md §4, 10_kernel_v2_design.md §5

v2: Sub-kernels declared as Kernel(), connections via >> operator,
    auto-expose, auto-forward chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    import torch

logger = logging.getLogger(__name__)

# ── Composite Registry ──
# Maps sub-kernel class → CompositeKernel class that contains it.
# Populated by CompositeKernel.__init_subclass__.
_composite_registry: dict[type, type] = {}


def _same_kernel_class(a: type, b: type) -> bool:
    """Check if two classes represent the same kernel, tolerating re-imports."""
    if a is b:
        return True
    if a.__name__ != b.__name__:
        return False
    import sys
    from pathlib import Path

    fa = getattr(sys.modules.get(a.__module__), "__file__", None)
    fb = getattr(sys.modules.get(b.__module__), "__file__", None)
    if fa and fb:
        return Path(fa).resolve() == Path(fb).resolve()
    return False


def _lookup_composite(kernel_cls: type) -> type | None:
    """Look up composite for kernel_cls, tolerating duplicate class objects.

    When the same .py file is loaded under different module names
    (e.g., ``_vten_kernel_act_quant`` vs ``act_quant.act_quant_kernel``),
    Python creates separate class objects.  Direct ``dict.get`` fails,
    so we fall back to matching by source file path.
    """
    # Fast path: exact identity match
    result = _composite_registry.get(kernel_cls)
    if result is not None:
        return result

    # Slow path: match by source file
    import sys

    src_mod = sys.modules.get(kernel_cls.__module__)
    src_file = getattr(src_mod, "__file__", None) if src_mod else None
    if src_file is None:
        return None

    from pathlib import Path

    src_path = Path(src_file).resolve()

    for reg_cls, comp_cls in _composite_registry.items():
        reg_mod = sys.modules.get(reg_cls.__module__)
        reg_file = getattr(reg_mod, "__file__", None) if reg_mod else None
        if reg_file and Path(reg_file).resolve() == src_path:
            # Cache for future lookups
            _composite_registry[kernel_cls] = comp_cls
            return comp_cls
    return None


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
    from vten.kernel.base import Kernel, RegisterHandle
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

            # Register sub-kernel → composite mapping
            for ref_class in cls._sub_kernel_refs.values():
                _composite_registry[ref_class] = cls

            # Auto-register {ref_name}_ctrl for each sub-kernel
            # (skip if user already declared it explicitly)
            for ref_name in cls._sub_kernel_refs:
                ctrl_name = f"{ref_name}_ctrl"
                if ctrl_name not in cls._register_handles:
                    handle = RegisterHandle(ctrl_name)
                    setattr(cls, ctrl_name, handle)
                    cls._register_handles[ctrl_name] = handle

        @classmethod
        def topo_sort_sub_kernels(cls) -> list[str]:
            """Return sub-kernel names in topological (dependency) order."""
            return _topo_sort(cls._connections, set(cls._sub_kernel_refs.keys()))

        def generate_inputs(self, seed: int | None = None) -> None:
            """Auto-delegate to source sub-kernels' generate_inputs().

            Identifies sub-kernels that have auto-exposed tensors and
            define their own generate_inputs(), then calls them in
            topo order.  Sub-kernel instances' tensors are shared with
            the composite, so exposed inputs get populated automatically.
            """
            cls = self.__class__
            has_exposed = {sub_name for (sub_name, _) in cls._auto_exposed}

            for sub_name in cls.topo_sort_sub_kernels():
                if sub_name not in has_exposed:
                    continue
                sub_inst = self._get_sub_kernel_instance(sub_name)
                if sub_inst is None:
                    continue
                sub_cls = type(sub_inst)
                if "generate_inputs" in sub_cls.__dict__:
                    sub_cls.generate_inputs(sub_inst, seed=seed)

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
            All data in the pool is physical (HW format).
            No layout/unlayout — that is the concern of the caller
            (generate_inputs, functional.py, etc.).
            """
            cls = self.__class__
            MAX_ROUNDS = 20

            pool: dict[tuple[str, str], Any] = {}

            # 1. Seed exposed input tensors from kwargs or instance data
            #    Auto-layout: if sub-kernel has layout_{name}(), apply it
            #    so forward() always receives physical data.
            for (sub_name, tensor_name), _t_name in cls._auto_exposed.items():
                exposed_key = f"{sub_name}_{tensor_name}"
                data = None
                if tensor_name in inputs:
                    data = inputs[tensor_name]
                elif exposed_key in inputs:
                    data = inputs[exposed_key]
                if data is not None:
                    sub_inst = self._get_sub_kernel_instance(sub_name)
                    if sub_inst is not None:
                        layout_fn = getattr(sub_inst, f"layout_{tensor_name}", None)
                        if layout_fn is not None and callable(layout_fn):
                            data = layout_fn(data)
                    pool[(sub_name, tensor_name)] = data

            # Fall back to sub-kernel instance tensor data
            for (sub_name, tensor_name) in cls._auto_exposed:
                if (sub_name, tensor_name) in pool:
                    continue
                sub_inst = self._get_sub_kernel_instance(sub_name)
                if sub_inst is not None:
                    t = sub_inst.get_tensor(tensor_name)
                    if t.data is not None:
                        layout_fn = getattr(sub_inst, f"layout_{tensor_name}", None)
                        if layout_fn is not None and callable(layout_fn):
                            pool[(sub_name, tensor_name)] = layout_fn(t.data)
                        else:
                            pool[(sub_name, tensor_name)] = t.data

            # 2. Multi-round dataflow evaluation
            sub_order = cls.topo_sort_sub_kernels()
            computed_outputs: set[tuple[str, str]] = set()

            for _round in range(MAX_ROUNDS):
                progress = False
                for sub_name in sub_order:
                    sub_inst = self._get_sub_kernel_instance(sub_name)
                    if sub_inst is None:
                        continue

                    # Collect available inputs for this sub-kernel
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

                    # Pass pool data directly to forward — no layout
                    forward_inputs: dict[str, Any] = {}
                    for name, data in available.items():
                        forward_inputs[name] = data
                    try:
                        outputs = sub_inst.forward(**forward_inputs)
                    except (NotImplementedError, TypeError):
                        continue

                    for out_name, out_data in outputs.items():
                        out_key = (sub_name, out_name)
                        if out_key not in computed_outputs:
                            pool[out_key] = out_data
                            computed_outputs.add(out_key)
                            progress = True
                            # Propagate through connections
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

            # 3. Collect exposed outputs
            result: dict[str, torch.Tensor] = {}
            for (sub_name, tensor_name) in cls._auto_exposed:
                key = (sub_name, tensor_name)
                if key in computed_outputs:
                    result[tensor_name] = pool[key]

            return result

        @classmethod
        def _generate_inputs_for(
            cls, target_kernel: Kernel, seed: int | None = None,
        ) -> None:
            """Generate inputs for target_kernel by running upstream chain.

            Walks the connection graph in topological order, creating
            temporary instances of upstream kernels, calling their
            generate_inputs() + forward(), and propagating outputs
            through connections until the target kernel's connected
            inputs are populated.

            Pool data is physical (HW format). Auto-layout is applied
            to exposed inputs before forward() calls.
            """
            target_cls = type(target_kernel)

            # Find which sub-ref name corresponds to target_cls
            # Use _same_kernel_class to handle duplicate class objects
            target_sub: str | None = None
            for name, kcls in cls._sub_kernel_refs.items():
                if _same_kernel_class(kcls, target_cls):
                    target_sub = name
                    break
            if target_sub is None:
                raise ValueError(
                    f"{target_cls.__name__} is not a sub-kernel of "
                    f"{cls.__name__}"
                )

            # Build execution order: only upstream dependencies of target,
            # in topological order.  The full topo sort may place the
            # target before some of its sources (cycles), so we compute
            # a targeted reverse-BFS from target_sub.
            needed: set[str] = set()

            def _collect_upstream(sub: str) -> None:
                for conn in cls._connections:
                    if conn.dest_sub == sub and conn.source_sub != target_sub:
                        if conn.source_sub not in needed:
                            needed.add(conn.source_sub)
                            _collect_upstream(conn.source_sub)

            _collect_upstream(target_sub)
            # Sort only upstream kernels (exclude target to avoid cycles)
            upstream_order = _topo_sort(cls._connections, needed) if needed else []
            order = upstream_order + [target_sub]
            logger.debug("auto-chain order=%s needed=%s target=%s", order, needed, target_sub)

            # Extract ALL resolved params from target for upstream instantiation.
            # Pass everything as runtime_params so upstream kernels get
            # project params, build params, and derived params.
            target_ki = getattr(target_kernel, "_kernel_instance", None)
            all_params: dict = {}
            if target_ki is not None and target_ki._resolver is not None:
                all_params = dict(target_ki._resolver.namespace)
            else:
                # No KernelInstance — collect scalar attrs from target
                for k, v in vars(target_kernel).items():
                    if not k.startswith("_") and isinstance(
                        v, (int, float, str),
                    ):
                        all_params[k] = v

            # Strategy: create a temporary composite, run generate_inputs,
            # then manually chain upstream forwards to populate target's
            # connected inputs.
            from vten.runtime.kernel_view import KernelInstance
            from vten.spec.models import KernelSpec

            comp_spec = KernelSpec(
                kernel_name=cls.__name__,
                rtl_top=cls.__name__,
            )
            comp_ki = KernelInstance(
                name="_tmp",
                spec=comp_spec,
                kernel_class=cls,
                runtime_params=dict(all_params),
            )
            comp_ki.initialize({})
            comp_inst = comp_ki.kernel_class_instance
            comp_inst._kernel_instance = comp_ki

            # Run composite generate_inputs (custom or auto-delegated)
            comp_inst.generate_inputs(seed=seed)

            # Chain upstream forwards and propagate via connections
            pool: dict[tuple[str, str], Any] = {}

            for sub_name in order:
                sub_inst = comp_inst._get_sub_kernel_instance(sub_name)
                if sub_inst is None:
                    continue

                if sub_name == target_sub:
                    # Set connected inputs on target kernel
                    for conn in cls._connections:
                        if conn.dest_sub == sub_name:
                            src_key = (conn.source_sub, conn.source_name)
                            if src_key in pool:
                                t = target_kernel.get_tensor(conn.dest_name)
                                t.data = pool[src_key]
                    # Copy exposed inputs from composite sub to target
                    for t in target_kernel.tensors():
                        if t.data is None:
                            src_t = sub_inst.get_tensor(t.name)
                            if src_t is not None and src_t.data is not None:
                                t.data = src_t.data
                    break

                # Collect inputs for upstream sub-kernel forward
                # Auto-layout: forward() expects physical data
                fwd_inputs: dict[str, Any] = {}
                for t in sub_inst.tensors():
                    if t.data is not None:
                        layout_fn = getattr(sub_inst, f"layout_{t.name}", None)
                        if layout_fn is not None and callable(layout_fn):
                            fwd_inputs[t.name] = layout_fn(t.data)
                        else:
                            fwd_inputs[t.name] = t.data
                # Also add connected data from pool
                for conn in cls._connections:
                    if conn.dest_sub == sub_name:
                        src_key = (conn.source_sub, conn.source_name)
                        if src_key in pool:
                            fwd_inputs[conn.dest_name] = pool[src_key]

                sub_cls = type(sub_inst)
                if "forward" not in sub_cls.__dict__:
                    continue
                try:
                    outputs = sub_cls.forward(sub_inst, **fwd_inputs)
                except Exception as e:
                    # Cycle-dependent nodes may fail (missing inputs,
                    # None tensors, etc.) — skip gracefully.
                    logger.debug("auto-chain: %s.forward() skipped: %s", sub_name, e)
                    continue
                for out_name, out_data in outputs.items():
                    pool[(sub_name, out_name)] = out_data

            n_copied = sum(1 for t in target_kernel.tensors() if t.data is not None)
            logger.debug(
                "generate_inputs_for: %s via %s, populated %d tensors",
                target_cls.__name__, cls.__name__, n_copied,
            )

        def get_tensor(self, name: str) -> Tensor:
            """Get a tensor by name, resolving auto-exposed sub-kernel tensors."""
            # Check auto_exposed: (sub_name, tensor_name) → exposed_name
            cls = self.__class__
            for (sub_name, tensor_name), exposed_name in cls._auto_exposed.items():
                if name == tensor_name or name == exposed_name:
                    sub = self._get_sub_kernel_instance(sub_name)
                    if sub is not None:
                        return sub.get_tensor(tensor_name)
            # Fall through to base class
            return super().get_tensor(name)

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
