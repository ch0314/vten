"""Stage 0: Kernel flattening — wraps unit or flattens composite kernels.

Extracted from RuntimeEngine to keep engine.py focused on pipeline orchestration.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from vten.errors import (
    ConnectionDtypeMismatchError,
    ProtocolMismatchError,
    ValidationError,
)
from vten.spec.models import (
    Direction,
    KernelSpec,
    MappingType,
    Protocol,
    Role,
)
from vten.runtime.kernel_view import (
    ExposedTensor,
    FlattenedKernelView,
    InterfaceMapping,
    KernelInstance,
    ProbePoint,
)

if TYPE_CHECKING:
    from vten.kernel.tensor import Tensor

logger = logging.getLogger(__name__)


# ── Direction inference ──


def infer_direction_unit(tensor: Tensor, spec: KernelSpec) -> Direction:
    """Resolve tensor direction: explicit value first, then protocol inference.

    Priority:
      1. tensor.direction (explicit) — use as-is
      2. AXI4-Stream — infer from interface role (master=H2D, slave=D2H)
      3. AXI4 (MM) — default HOST_TO_DEV with warning
      4. Fallback — HOST_TO_DEV
    """
    if tensor.direction is not None:
        return tensor.direction

    try:
        iface = spec.get_interface(tensor.interface)
    except KeyError:
        return Direction.HOST_TO_DEV

    if iface.protocol == Protocol.AXI4S:
        if hasattr(iface, "role") and iface.role == Role.SLAVE:
            return Direction.DEV_TO_HOST
        return Direction.HOST_TO_DEV

    if iface.protocol == Protocol.AXI4:
        warnings.warn(
            f"Tensor '{tensor.name}' on AXI4 interface "
            f"'{tensor.interface}' has no explicit direction; "
            f"defaulting to HOST_TO_DEV. Consider setting "
            f"direction=Direction.HOST_TO_DEV or "
            f"direction=Direction.DEV_TO_HOST explicitly.",
            stacklevel=3,
        )
        return Direction.HOST_TO_DEV

    return Direction.HOST_TO_DEV


def infer_direction_composite(
    sub_kernel_name: str,
    tensor: Tensor,
    mappings: list[InterfaceMapping],
    sub_spec: KernelSpec,
) -> Direction:
    """Infer tensor direction for CompositeKernel exposed tensors."""
    if tensor.direction is not None:
        return tensor.direction
    return infer_direction_unit(tensor, sub_spec)


# ── Flatten ──


def is_composite(kernel: KernelInstance) -> bool:
    """Check if a KernelInstance wraps a CompositeKernel."""
    return bool(getattr(kernel.kernel_class, "_sub_kernel_refs", None))


def wrap_unit_as_flat(kernel: KernelInstance) -> FlattenedKernelView:
    """Wrap a Unit kernel as FlattenedKernelView with _self sub-kernel."""
    mappings: list[InterfaceMapping] = []
    for iface in kernel.spec.interfaces.values():
        mappings.append(
            InterfaceMapping(
                sub_kernel="_self",
                sub_interface=iface.name,
                mapping_type=MappingType.EXTERNAL,
                top_interface=iface.name,
                bank_name=None,
                bank_offset=0,
            )
        )

    exposed: dict[str, ExposedTensor] = {}
    for tensor in kernel.tensors():
        direction = infer_direction_unit(tensor, kernel.spec)
        exposed[tensor.name] = ExposedTensor(
            name=tensor.name,
            origin_path=f"_self.{tensor.name}",
            origin_tensor=tensor,
            top_interface=tensor.interface,
            direction=direction,
        )

    return FlattenedKernelView(
        name=kernel.name,
        top_spec=kernel.spec,
        sub_kernels={"_self": kernel},
        interface_mappings=mappings,
        exposed_tensors=exposed,
        probe_points=[],
        connections=[],
    )


def flatten_composite(
    kernel: KernelInstance,
    project_params: dict,
    *,
    project_dir: Path | None = None,
) -> FlattenedKernelView:
    """Flatten a CompositeKernel into FlattenedKernelView.

    v2: Uses _sub_kernel_refs, auto-expose, auto-prefix registers.
    No more interface_map, config_map, bind(), expose(), Internal().
    """
    from vten.kernel.composite import Connection
    from vten.spec.parser import load_kernel_spec

    composite_instance = kernel.kernel_class_instance
    composite_cls = kernel.kernel_class
    top_spec = kernel.spec

    # ── Phase A: Sub-kernel instantiation ──
    existing_subs = kernel._sub_kernel_instances
    sub_kernels: dict[str, KernelInstance] = {}
    sub_kernel_refs = getattr(composite_cls, "_sub_kernel_refs", {})

    import os
    from pathlib import Path as _P
    _proj_dir = project_dir if project_dir is not None else _P(os.getcwd())

    for ref_name, sub_cls in sub_kernel_refs.items():
        if existing_subs and ref_name in existing_subs:
            sub_ki = existing_subs[ref_name]
            sub_spec_path = getattr(sub_cls, "spec", "")
            if sub_spec_path and not sub_ki.spec.interfaces:
                resolved = _proj_dir / sub_spec_path if not _P(sub_spec_path).is_absolute() else _P(sub_spec_path)
                sub_ki.spec = load_kernel_spec(resolved)
        else:
            sub_spec_path = getattr(sub_cls, "spec", "")
            if sub_spec_path:
                resolved = _proj_dir / sub_spec_path if not _P(sub_spec_path).is_absolute() else _P(sub_spec_path)
                sub_spec = load_kernel_spec(resolved)
            else:
                sub_spec = KernelSpec(
                    kernel_name=sub_cls.__name__,
                    rtl_top=sub_cls.__name__,
                )
            sub_ki = KernelInstance(
                name=ref_name,
                spec=sub_spec,
                kernel_class=sub_cls,
                runtime_params=dict(kernel.runtime_params),
            )
            sub_ki.initialize(project_params, project_dir=project_dir)
        sub_kernels[ref_name] = sub_ki

    # Synthesize top-level spec for composite kernels if missing
    if not top_spec.interfaces:
        cached = getattr(kernel.kernel_class, "_synthesized_spec", None)
        if cached is not None:
            top_spec = cached
        else:
            import os
            from vten.build.composite import synthesize_spec
            _synth_dir = project_dir if project_dir is not None else Path(os.getcwd())
            top_spec = synthesize_spec(
                kernel.kernel_class, _synth_dir, kernel.name,
            )
            kernel.kernel_class._synthesized_spec = top_spec
        kernel.spec = top_spec

    # ── Phase B: Interface mapping (auto-inferred) ──
    connections = getattr(composite_cls, "_connections", []) or []
    connected_tensors = getattr(composite_cls, "_connected_tensors", set())
    auto_exposed = getattr(composite_cls, "_auto_exposed", {})
    mappings: list[InterfaceMapping] = []

    for ref_name, sub_ki in sub_kernels.items():
        sub_spec = sub_ki.spec
        for sub_iface_name in sub_spec.interface_names():
            # Check if this interface's tensors are connected (internal)
            is_internal = False
            for t_name, t_desc in sub_ki.kernel_class._tensor_descriptors.items():
                if t_desc.interface == sub_iface_name:
                    if (ref_name, t_name) in connected_tensors:
                        is_internal = True
                        break

            if is_internal:
                # Internal connection → INTERNAL_PROBE (always probe-capable)
                mappings.append(InterfaceMapping(
                    sub_kernel=ref_name,
                    sub_interface=sub_iface_name,
                    mapping_type=MappingType.INTERNAL_PROBE,
                    top_interface=None,
                    bank_name=None,
                    bank_offset=0,
                ))
            else:
                # Auto-expose → EXTERNAL with auto-prefix
                top_iface = f"{ref_name}_{sub_iface_name}"
                mappings.append(InterfaceMapping(
                    sub_kernel=ref_name,
                    sub_interface=sub_iface_name,
                    mapping_type=MappingType.EXTERNAL,
                    top_interface=top_iface,
                    bank_name=None,
                    bank_offset=0,
                ))

    # ── Phase C: Auto-exposed tensor collection ──
    exposed: dict[str, ExposedTensor] = {}

    for (sub_name, tensor_name), _t_name in auto_exposed.items():
        sub_ki = sub_kernels[sub_name]
        origin_tensor = sub_ki.get_tensor(tensor_name)
        direction = infer_direction_composite(
            sub_name, origin_tensor, mappings, sub_ki.spec
        )
        # Find top_interface for this tensor
        top_iface = f"{sub_name}_{origin_tensor.interface}"
        # Key by tensor_name for lookup by op.tensor.name;
        # if conflict (same tensor_name from different sub-kernels),
        # fall back to prefixed name
        exposed_name = tensor_name
        if exposed_name in exposed:
            exposed_name = f"{sub_name}_{tensor_name}"
        exposed[exposed_name] = ExposedTensor(
            name=exposed_name,
            origin_path=f"{sub_name}.{tensor_name}",
            origin_tensor=origin_tensor,
            top_interface=top_iface,
            direction=direction,
        )

    # ── Phase D: Probe point collection ──
    probe_points: list[ProbePoint] = []
    probed_ifaces: set[tuple[str, str]] = set()
    probe_mapping_by_key: dict[tuple[str, str], InterfaceMapping] = {}
    for m in mappings:
        if m.mapping_type == MappingType.INTERNAL_PROBE:
            key = (m.sub_kernel, m.sub_interface)
            probed_ifaces.add(key)
            probe_mapping_by_key[key] = m

    for conn in connections:
        src_key = (conn.source_sub, conn.source_interface)
        dst_iface = conn.dest_interface
        dst_key = (conn.dest_sub, dst_iface) if dst_iface else None

        matched_key = None
        if src_key in probed_ifaces:
            matched_key = src_key
        elif dst_key is not None and dst_key in probed_ifaces:
            matched_key = dst_key

        if matched_key is not None:
            m = probe_mapping_by_key[matched_key]
            probe_points.append(
                ProbePoint(connection=conn, interface_mapping=m)
            )

    # ── Phase E: Validation ──
    validate_flattened(
        mappings, exposed, connections, top_spec, sub_kernels
    )

    view = FlattenedKernelView(
        name=kernel.name,
        top_spec=top_spec,
        sub_kernels=sub_kernels,
        interface_mappings=mappings,
        exposed_tensors=exposed,
        probe_points=probe_points,
        connections=connections,
    )
    return view


# ── Validation ──


def validate_flattened(
    mappings: list[InterfaceMapping],
    exposed: dict[str, ExposedTensor],
    connections: list,
    top_spec: KernelSpec,
    sub_kernels: dict[str, KernelInstance] | None = None,
) -> None:
    """Build-time validation of flattened view."""
    external_ifaces = {
        m.top_interface
        for m in mappings
        if m.mapping_type in (MappingType.EXTERNAL, MappingType.EXTERNAL_BANK)
        and m.top_interface is not None
    }
    for name, exp in exposed.items():
        if exp.top_interface not in external_ifaces:
            raise ValidationError(
                f"ExposedTensor '{name}' maps to top_interface "
                f"'{exp.top_interface}' which has no EXTERNAL mapping."
            )

    if not connections or not sub_kernels:
        return

    _validate_connection_protocols(connections, sub_kernels)
    _validate_connection_dtypes(connections, sub_kernels)
    _validate_internal_coverage(mappings, connections, sub_kernels)
    _validate_no_duplicate_connections(connections, sub_kernels)


def _validate_connection_protocols(
    connections: list,
    sub_kernels: dict[str, KernelInstance],
) -> None:
    """Validate that connected interfaces use the same protocol."""
    for conn in connections:
        src_ki = sub_kernels.get(conn.source_sub)
        dst_ki = sub_kernels.get(conn.dest_sub)
        if not src_ki or not dst_ki:
            continue
        src_iface = src_ki.spec.interfaces.get(conn.source_interface)
        dest_iface_name = getattr(conn, "dest_interface", None)
        if not dest_iface_name:
            continue
        dst_iface = dst_ki.spec.interfaces.get(dest_iface_name)
        if src_iface and dst_iface and src_iface.protocol != dst_iface.protocol:
            raise ProtocolMismatchError(
                f"Connection {conn.source_sub}.{conn.source_interface} "
                f"({src_iface.protocol.value}) → "
                f"{conn.dest_sub}.{dest_iface_name} "
                f"({dst_iface.protocol.value}): protocol mismatch"
            )


def _validate_connection_dtypes(
    connections: list,
    sub_kernels: dict[str, KernelInstance],
) -> None:
    """Validate that connected tensors have matching dtype.

    For internal (RTL wire) connections, dtype mismatch is allowed
    because the physical bus carries raw bytes regardless of the
    logical dtype declared on each tensor.
    """
    for conn in connections:
        if getattr(conn, "is_internal_wire", True):
            continue
        src_ki = sub_kernels.get(conn.source_sub)
        dst_ki = sub_kernels.get(conn.dest_sub)
        if not src_ki or not dst_ki:
            continue
        try:
            src_tensor = src_ki.get_tensor(conn.source_name)
            dst_tensor = dst_ki.get_tensor(conn.dest_name)
        except (RuntimeError, AttributeError):
            continue
        if (
            src_tensor.dtype != dst_tensor.dtype
            and conn.transform is None
        ):
            raise ConnectionDtypeMismatchError(
                f"Connection {conn.source_sub}.{conn.source_name} "
                f"(dtype={src_tensor.dtype}) → "
                f"{conn.dest_sub}.{conn.dest_name} "
                f"(dtype={dst_tensor.dtype}): "
                f"dtype mismatch without explicit transform"
            )


def _validate_internal_coverage(
    mappings: list[InterfaceMapping],
    connections: list,
    sub_kernels: dict[str, KernelInstance],
) -> None:
    """Validate that all Internal() interfaces are covered by connections."""
    internal_ifaces: set[tuple[str, str]] = set()
    for m in mappings:
        if m.mapping_type in (MappingType.INTERNAL, MappingType.INTERNAL_PROBE):
            internal_ifaces.add((m.sub_kernel, m.sub_interface))

    probe_ifaces = {
        (m.sub_kernel, m.sub_interface)
        for m in mappings
        if m.mapping_type == MappingType.INTERNAL_PROBE
    }
    must_connect = internal_ifaces - probe_ifaces

    connected: set[tuple[str, str]] = set()
    for conn in connections:
        connected.add((conn.source_sub, conn.source_interface))
        dest_iface_name = getattr(conn, "dest_interface", None)
        if dest_iface_name:
            connected.add((conn.dest_sub, dest_iface_name))

    dangling = must_connect - connected
    if dangling:
        dangling_desc = ", ".join(
            f"{sub}.{iface}" for sub, iface in sorted(dangling)
        )
        raise ValidationError(
            f"Internal interfaces have no connection: {dangling_desc}"
        )


def _validate_no_duplicate_connections(
    connections: list,
    sub_kernels: dict[str, KernelInstance],
) -> None:
    """Validate no interface appears in multiple connections."""
    seen_src: set[tuple[str, str]] = set()
    seen_dst: set[tuple[str, str]] = set()
    for conn in connections:
        src_key = (conn.source_sub, conn.source_interface)
        if src_key in seen_src:
            raise ValidationError(
                f"Duplicate connection source: "
                f"{conn.source_sub}.{conn.source_interface}"
            )
        seen_src.add(src_key)

        dest_iface_name = getattr(conn, "dest_interface", None)
        if dest_iface_name:
            dst_key = (conn.dest_sub, dest_iface_name)
            if dst_key in seen_dst:
                raise ValidationError(
                    f"Duplicate connection destination: "
                    f"{conn.dest_sub}.{dest_iface_name}"
                )
            seen_dst.add(dst_key)
