"""RuntimeEngine — 8-stage compile pipeline orchestrator.

Spec reference: 02_runtime_engine.md §4
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from vten.errors import (
    CompilationError,
    ConnectionDtypeMismatchError,
    ConnectionShapeMismatchError,
    ProtocolMismatchError,
    SerializationError,
    ShapeMismatchError,
    ValidationError,
)
from vten.kernel.tensor import Tensor
from vten.spec.models import (
    Direction,
    KernelSpec,
    MappingType,
    OpCode,
    Protocol,
    Role,
)
from vten.runtime.address import AddressAllocator
from vten.runtime.binder import (
    RegisterBindingEntry,
    resolve_registers,
)
from vten.runtime.flattener import (
    ExposedTensor,
    FlattenedKernelView,
    InterfaceMapping,
    KernelInstance,
    ProbePoint,
)
from vten.runtime.ir import BFMConfig, Command, IRLowering, _determine_role
from vten.runtime.resolver import ParameterResolver
from vten.runtime.serializer import MultiPortSerializer, StreamSerializer
from vten.runtime.shm_packer import (
    _block_split_data,
    _parse_split_spec,
    SHMLayout,
    pack_shm,
    pack_shm_multi,
)

if TYPE_CHECKING:
    from vten.dsl.operations import Operation
    from vten.runtime.context import AliasRegistry


# ── CompiledResult ──


@dataclass
class CompiledResult:
    commands: list[Command]
    shm_image: bytes
    bfm_configs: list[BFMConfig]
    buffer_ids: dict[str, int]
    flattened_view: FlattenedKernelView
    ops: list[Operation] = field(default_factory=list)  # original DSL operations
    probe_reports: list[ProbePoint] = field(default_factory=list)
    tensor_data: dict[int, bytes] = field(default_factory=dict)
    iface_id_to_name: dict[int, str] = field(default_factory=dict)
    shm_layout: SHMLayout | None = None
    views: list[FlattenedKernelView] | None = None  # multi-config: all views
    probe_buffer_map: dict[int, int] = field(default_factory=dict)  # probe_index → golden_buffer_id
    prebound_buffers: dict[int, object] = field(default_factory=dict)  # buffer_id → xrt.bo (inference)
    mode: str = "verification"  # "verification" or "inference"


# ── RuntimeEngine ──


class RuntimeEngine:
    """8-stage compile pipeline orchestrator."""

    def __init__(
        self,
        kernels: dict[str, KernelInstance],
        ops: list[Operation],
        project_params: dict,
        alias_registry: AliasRegistry | None = None,
        quiet: bool = False,
    ) -> None:
        self._kernels = kernels
        self._ops = ops
        self._project_params = project_params
        self._alias_registry = alias_registry
        self._quiet = quiet

    # ── Internal: Stages 0–6 (IR generation, no SHM packing) ──

    def _compile_ir(
        self,
        *,
        cmd_id_start: int = 0,
        buffer_id_start: int = 0,
    ) -> tuple[
        FlattenedKernelView,
        list[Command],
        dict[str, int],
        list[BFMConfig],
        dict[int, str],
    ]:
        """Run Stages 0–6 and return intermediate results.

        Returns:
            (view, commands, buffer_ids, bfm_configs, iface_id_to_name)
        """
        kernel = self._get_primary_kernel()

        # Stage 0: Flatten or wrap
        logger.log(5, "Stage 0: flatten/wrap")
        if self._is_composite(kernel):
            view = self._flatten_composite(kernel)
        else:
            view = self._wrap_unit_as_flat(kernel)

        # Stage 1: Parameter resolution (re-validate)
        logger.log(5, "Stage 1: parameter resolution")
        self._resolve_parameters(view)

        # Stage 2: Shape resolution & validation
        logger.log(5, "Stage 2: shape resolution")
        self._resolve_shapes(view)

        # Stage 2b: Refine direction from operations
        logger.log(5, "Stage 2b: direction refinement")
        self._refine_directions_from_ops(view)

        # Stage 3: Tensor serialization
        logger.log(5, "Stage 3: tensor serialization")
        self._serialize_tensors(view)

        # Stage 3b: Probe golden serialization
        logger.log(5, "Stage 3b: probe golden serialization")
        self._serialize_probe_golden(view)

        # Stage 4: Address allocation
        logger.log(5, "Stage 4: address allocation")
        self._allocate_addresses(view)

        # Stage 5: auto_bind resolution
        logger.log(5, "Stage 5: auto_bind resolution")
        self._resolve_registers(view)

        # Stage 6: IR lowering
        logger.log(5, "Stage 6: IR lowering")
        lowering = IRLowering(view, self._alias_registry)
        commands, buffer_ids = lowering.lower(
            self._ops,
            cmd_id_start=cmd_id_start,
            buffer_id_start=buffer_id_start,
        )

        iface_id_to_name: dict[int, str] = {
            v: k for k, v in lowering._iface_id_map.items()
        }

        self._log_ir_summary(commands, quiet=self._quiet)
        self._log_dataflow_summary(view, quiet=self._quiet)
        if logger.isEnabledFor(logging.DEBUG):
            self._log_ir_commands(commands, iface_id_to_name, buffer_ids)

        # Stage 6b: BFM configuration synthesis
        logger.log(5, "Stage 6b: BFM config synthesis")
        bfm_configs = self._synthesize_bfm_configs(view, commands, buffer_ids)

        return view, commands, buffer_ids, bfm_configs, iface_id_to_name

    @staticmethod
    def _collect_tensor_data(
        view: FlattenedKernelView,
        buffer_ids: dict[str, int],
    ) -> dict[int, bytes]:
        """Collect serialized tensor data from a flattened view."""
        tensor_data: dict[int, bytes] = {}
        for name, exposed in view.exposed_tensors.items():
            if exposed._port_buffers:
                for flat_name, chunk in exposed._port_buffers.items():
                    bid = buffer_ids.get(f"{name}:{flat_name}")
                    if bid is not None and chunk:
                        tensor_data[bid] = chunk
            elif exposed._serialized is not None:
                bid = buffer_ids.get(name)
                if bid is not None:
                    tensor_data[bid] = exposed._serialized
        return tensor_data

    # ── Public: Single-config compile ──

    def compile(
        self,
        target: str = "sim",
        probe_golden_tensors: dict | None = None,
        internal_probe_golden: dict | None = None,
        flags: int = 0,
    ) -> CompiledResult:
        """Run the 8-stage compile pipeline.

        Args:
            target: "sim" for SIM backends (includes Stage 7 SHM packing),
                    "hw" for HW backends (skips SHM packing).
            probe_golden_tensors: tensor_name → torch.Tensor golden data for probe.
            internal_probe_golden: (sub_name, tensor_name) → torch.Tensor for
                composite internal probe golden data.
            flags: SHM control header flags (e.g. FLAG_WAVEFORM_DUMP).
        """
        t0 = time.perf_counter()
        logger.log(5, "compile pipeline starting (target=%s, ops=%d)", target, len(self._ops))

        view, commands, buffer_ids, bfm_configs, iface_id_to_name = (
            self._compile_ir()
        )

        # Populate view.probe_points for single-kernel probe golden tensors
        if probe_golden_tensors:
            self._add_probe_golden_to_view(view, probe_golden_tensors)

        # Ensure INTERNAL → INTERNAL_PROBE upgrade for declarative probes
        if internal_probe_golden:
            self._ensure_probe_mappings(view, internal_probe_golden)
            self._serialize_probe_golden(view, internal_probe_golden)

        # Stage 7: SHM packing (SIM path only)
        probe_buffer_map: dict[int, int] = {}
        shm_layout: SHMLayout | None = None
        if target == "sim":
            logger.log(5, "Stage 7: SHM packing")
            shm_image, shm_layout = pack_shm(view, commands, buffer_ids, self._ops, flags=flags)
            logger.log(5, "SHM image: %d bytes", len(shm_image))
            # Build probe_index → golden_buffer_id mapping for plusargs
            probe_buffer_map = self._build_probe_buffer_map(view)
        else:
            shm_image = b""

        tensor_data = self._collect_tensor_data(view, buffer_ids)

        elapsed = time.perf_counter() - t0
        logger.debug("compile complete: %d commands, %d buffers, %.1fms",
                     len(commands), len(buffer_ids), elapsed * 1000)

        return CompiledResult(
            commands=commands,
            shm_image=shm_image,
            bfm_configs=bfm_configs,
            buffer_ids=buffer_ids,
            flattened_view=view,
            ops=list(self._ops),
            probe_reports=view.probe_points,
            tensor_data=tensor_data,
            iface_id_to_name=iface_id_to_name,
            shm_layout=shm_layout,
            probe_buffer_map=probe_buffer_map,
            mode="inference" if self._quiet else "verification",
        )

    @staticmethod
    def _log_ir_commands(
        commands: list,
        iface_map: dict[int, str],
        buffer_ids: dict[str, int],
    ) -> None:
        """Log compiled IR commands as a readable table."""
        # Reverse buffer_ids: id → name
        bid_to_name: dict[int, str] = {v: k for k, v in buffer_ids.items()}

        lines: list[str] = ["compiled IR commands:"]
        lines.append(
            f"  {'#':>3}  {'Op':<10} {'Iface':<16} {'Buf':<16} "
            f"{'Size':>8}  {'Dep':<10} {'Extra'}"
        )
        lines.append("  " + "-" * 78)

        for cmd in commands:
            op = cmd.op.name
            iface = iface_map.get(cmd.interface_id, str(cmd.interface_id))
            buf = bid_to_name.get(cmd.buffer_id, str(cmd.buffer_id))
            size = cmd.size
            dep = ",".join(str(d) for d in cmd.dep) if cmd.dep else "-"

            # Extra info depends on op type
            extra_parts: list[str] = []
            if cmd.reg_offset:
                extra_parts.append(f"reg=0x{cmd.reg_offset:X}")
            if cmd.reg_value:
                extra_parts.append(f"val=0x{cmd.reg_value:X}")
            if cmd.reg_mask:
                extra_parts.append(f"mask=0x{cmd.reg_mask:X}")
            if cmd.probe:
                extra_parts.append("probe")
            if cmd.sync:
                extra_parts.append("sync")
            if cmd.port:
                extra_parts.append(f"port={cmd.port}")
            extra = " ".join(extra_parts)

            lines.append(
                f"  {cmd.cmd_id:3d}  {op:<10} {iface:<16} {buf:<16} "
                f"{size:>8}  {dep:<10} {extra}"
            )

        logger.debug("\n".join(lines))

    @staticmethod
    def _log_ir_summary(commands: list, quiet: bool = False) -> None:
        """Log a concise IR summary. DEBUG when quiet (inference mode)."""
        from vten.log import format_size
        _log = logger.debug if quiet else logger.info
        n_cfg = sum(1 for c in commands if c.op == OpCode.WRITE_REG)
        n_load = sum(1 for c in commands if c.op == OpCode.LOAD)
        n_push = sum(1 for c in commands if c.op == OpCode.PUSH)
        n_pull = sum(1 for c in commands if c.op == OpCode.PULL)
        n_store = sum(1 for c in commands if c.op == OpCode.STORE)
        n_poll = sum(1 for c in commands if c.op == OpCode.POLL_REG)
        xfer_bytes = sum(c.size for c in commands if c.op == OpCode.LOAD and c.size)
        recv_bytes = sum(c.size for c in commands if c.op == OpCode.PULL and c.size)
        parts = []
        if n_cfg:
            parts.append(f"{n_cfg} reg")
        if n_load + n_push:
            parts.append(f"{n_load}+{n_push} xfer {format_size(xfer_bytes)}")
        if n_poll:
            parts.append(f"{n_poll} poll")
        if n_pull + n_store:
            parts.append(f"{n_pull}+{n_store} recv {format_size(recv_bytes)}")
        _log("compiled: %d cmds (%s)", len(commands), ", ".join(parts))

    @staticmethod
    def _log_dataflow_summary(view: FlattenedKernelView, quiet: bool = False) -> None:
        """Log inter-kernel dataflow. DEBUG when quiet (inference mode)."""
        if not view.connections:
            return  # Unit kernel — no dataflow to report

        from vten.log import format_size
        from vten.spec.models import Direction

        _log = logger.debug if quiet else logger.info
        lines = ["dataflow:"]

        # Internal connections (sub → sub)
        for conn in view.connections:
            src = f"{conn.source_sub}.{conn.source_name}"
            dst = f"{conn.dest_sub}.{conn.dest_name}"
            lines.append(f"  {src} → {dst}")

        # Exposed tensors (host ↔ device)
        host_to_dev = []
        dev_to_host = []
        for name, exp in view.exposed_tensors.items():
            size_str = ""
            if exp._serialized_size:
                size_str = f" {format_size(exp._serialized_size)}"
            label = f"{name}{size_str}"
            if exp.direction == Direction.HOST_TO_DEV:
                host_to_dev.append(label)
            elif exp.direction == Direction.DEV_TO_HOST:
                dev_to_host.append(label)

        if host_to_dev:
            lines.append(f"  host → device: {', '.join(host_to_dev)}")
        if dev_to_host:
            lines.append(f"  device → host: {', '.join(dev_to_host)}")

        _log("\n".join(lines))

    def _get_primary_kernel(self) -> KernelInstance:
        if not self._kernels:
            raise CompilationError("No kernels instantiated")
        return next(iter(self._kernels.values()))

    # ── Stage 0: Flattening ──

    def _wrap_unit_as_flat(self, kernel: KernelInstance) -> FlattenedKernelView:
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
            direction = self._infer_direction_unit(tensor, kernel.spec)
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

    def _is_composite(self, kernel: KernelInstance) -> bool:
        """Check if a KernelInstance wraps a CompositeKernel."""
        return bool(getattr(kernel.kernel_class, "_sub_kernel_refs", None))

    def _flatten_composite(self, kernel: KernelInstance) -> FlattenedKernelView:
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
        _proj_dir = _P(
            self._project_params.get("_project_dir", os.getcwd())
        )

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
                sub_ki.initialize(self._project_params)
            sub_kernels[ref_name] = sub_ki

        # Synthesize top-level spec for composite kernels if missing
        if not top_spec.interfaces:
            cached = getattr(kernel.kernel_class, "_synthesized_spec", None)
            if cached is not None:
                top_spec = cached
            else:
                import os
                from pathlib import Path as _Path
                from vten.build.composite import synthesize_spec
                project_dir = _Path(
                    self._project_params.get("_project_dir", os.getcwd())
                )
                top_spec = synthesize_spec(
                    kernel.kernel_class, project_dir, kernel.name,
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
            direction = self._infer_direction_composite(
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
        self._validate_flattened(
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

    # _parse_mapping removed in v2 (auto-inferred interface mappings)

    def _infer_direction_composite(
        self,
        sub_kernel_name: str,
        tensor: Tensor,
        mappings: list[InterfaceMapping],
        sub_spec: KernelSpec,
    ) -> Direction:
        """Infer tensor direction for CompositeKernel exposed tensors."""
        if tensor.direction is not None:
            return tensor.direction
        # Fall back to unit inference using sub-kernel spec
        return self._infer_direction_unit(tensor, sub_spec)

    # _find_connection_for_interface and _find_dest_interface_name
    # removed in v2 (Connection has direct .dest_interface property)

    def _validate_flattened(
        self,
        mappings: list[InterfaceMapping],
        exposed: dict[str, ExposedTensor],
        connections: list,
        top_spec: KernelSpec,
        sub_kernels: dict[str, KernelInstance] | None = None,
    ) -> None:
        """Build-time validation of flattened view."""
        # Validate all exposed tensors reference valid top interfaces
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

        # Connection validations (composite only)
        if not connections or not sub_kernels:
            return

        self._validate_connection_protocols(connections, sub_kernels)
        self._validate_connection_dtypes(connections, sub_kernels)
        self._validate_internal_coverage(mappings, connections, sub_kernels)
        self._validate_no_duplicate_connections(connections, sub_kernels)

    def _validate_connection_protocols(
        self,
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
        self,
        connections: list,
        sub_kernels: dict[str, KernelInstance],
    ) -> None:
        """Validate that connected tensors have matching dtype.

        For internal (RTL wire) connections, dtype mismatch is allowed
        because the physical bus carries raw bytes regardless of the
        logical dtype declared on each tensor.
        """
        for conn in connections:
            # Internal wires pass physical bytes — dtype semantics don't apply
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
        self,
        mappings: list[InterfaceMapping],
        connections: list,
        sub_kernels: dict[str, KernelInstance],
    ) -> None:
        """Validate that all Internal() interfaces are covered by connections."""
        internal_ifaces: set[tuple[str, str]] = set()
        for m in mappings:
            if m.mapping_type in (MappingType.INTERNAL, MappingType.INTERNAL_PROBE):
                internal_ifaces.add((m.sub_kernel, m.sub_interface))

        # Probe interfaces don't need connections
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
        self,
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

    def _infer_direction_unit(self, tensor, spec) -> Direction:
        """Resolve tensor direction: explicit value first, then protocol inference.

        Priority:
          1. tensor.direction (explicit) — use as-is
          2. AXI4-Stream — infer from interface role (master=H2D, slave=D2H)
          3. AXI4 (MM) — default HOST_TO_DEV with warning
          4. Fallback — HOST_TO_DEV
        """
        # Step 1: Explicit direction takes precedence
        if tensor.direction is not None:
            return tensor.direction

        # Step 2: Protocol-based inference
        try:
            iface = spec.get_interface(tensor.interface)
        except KeyError:
            return Direction.HOST_TO_DEV

        if iface.protocol == Protocol.AXI4S:
            if hasattr(iface, "role") and iface.role == Role.SLAVE:
                return Direction.DEV_TO_HOST
            return Direction.HOST_TO_DEV

        if iface.protocol == Protocol.AXI4:
            # AXI4 MM: same port can have read/write tensors.
            # Without explicit direction, default H2D with warning.
            import warnings
            warnings.warn(
                f"Tensor '{tensor.name}' on AXI4 interface "
                f"'{tensor.interface}' has no explicit direction; "
                f"defaulting to HOST_TO_DEV. Consider setting "
                f"direction=Direction.HOST_TO_DEV or "
                f"direction=Direction.DEV_TO_HOST explicitly.",
                stacklevel=2,
            )
            return Direction.HOST_TO_DEV

        return Direction.HOST_TO_DEV

    # ── Stage 1: Parameter Resolution ──

    def _resolve_parameters(self, view: FlattenedKernelView) -> None:
        """Re-validate parameters (already resolved during instantiate)."""
        for name, sub in view.sub_kernels.items():
            if sub._resolver is None:
                sub._resolver = ParameterResolver(
                    self._project_params,
                    sub.spec.parameters,
                    sub.runtime_params,
                )
        if view.sub_kernels:
            first_sub = next(iter(view.sub_kernels.values()))
            view._top_resolver = first_sub._resolver

    # ── Stage 2: Shape Resolution & Validation ──

    def _resolve_shapes(self, view: FlattenedKernelView) -> None:
        for name, sub in view.sub_kernels.items():
            for tensor in sub.tensors():
                if tensor._resolved_shape is None and sub._resolver:
                    tensor._resolve_shape(sub._resolver)

                if tensor.data is not None:
                    actual = tensor.data.numel()
                    if actual != tensor._element_count:
                        raise ShapeMismatchError(
                            f"Tensor '{name}.{tensor.name}': declared shape "
                            f"{tensor._resolved_shape} ({tensor._element_count} "
                            f"elements) but data has {actual} elements."
                        )

        # Connection shape compatibility (Composite only)
        # Skip for internal RTL wires — tensor shapes are for host
        # serialization and may intentionally differ between sub-kernels.
        for conn in view.connections:
            if getattr(conn, "is_internal_wire", False):
                continue
            src = view.sub_kernels[conn.source_sub].get_tensor(conn.source_name)
            dst = view.sub_kernels[conn.dest_sub].get_tensor(conn.dest_name)
            if src._element_count != dst._element_count:
                raise ConnectionShapeMismatchError(
                    f"Connection {conn.source_sub}.{conn.source_name} → "
                    f"{conn.dest_sub}.{conn.dest_name}: "
                    f"{src._element_count} vs {dst._element_count} elements."
                )

    # ── Stage 2b: Direction Refinement from Operations ──

    def _refine_directions_from_ops(self, view: FlattenedKernelView) -> None:
        """Refine ExposedTensor.direction using actual DSL operations.

        For tensors without explicit direction (defaulted to HOST_TO_DEV
        in Stage 0), override based on how they're used:
          send_tensor / push_tensor / load_tensor → HOST_TO_DEV
          recv_tensor / pull_tensor / store_tensor → DEV_TO_HOST

        Tensors not referenced by any operation but with no data are
        assumed to be output buffers (DEV_TO_HOST).
        """
        from vten.spec.models import OpKind

        h2d_ops = {OpKind.PUSH_TENSOR}
        d2h_ops = {OpKind.PULL_TENSOR}

        referenced: set[str] = set()
        for op in self._ops:
            if op.tensor is None:
                continue
            tensor_name = op.tensor.name
            if tensor_name not in view.exposed_tensors:
                continue
            referenced.add(tensor_name)
            exposed = view.exposed_tensors[tensor_name]
            if op.kind in d2h_ops:
                exposed.direction = Direction.DEV_TO_HOST
            elif op.kind in h2d_ops:
                exposed.direction = Direction.HOST_TO_DEV

        # Unreferenced tensors without data and without explicit direction:
        # assume they are output buffers
        for name, exposed in view.exposed_tensors.items():
            if name not in referenced and exposed.origin_tensor.direction is None:
                if exposed.origin_tensor.data is None:
                    exposed.direction = Direction.DEV_TO_HOST

    # ── Stage 3: Tensor Serialization ──

    def _serialize_tensors(self, view: FlattenedKernelView) -> None:
        for name, exposed in view.exposed_tensors.items():
            try:
                iface_spec = view.top_spec.get_interface(exposed.top_interface)
            except KeyError:
                continue

            packing = iface_spec.packing
            if packing is None:
                continue

            if exposed.direction == Direction.HOST_TO_DEV:
                # Zero-element tensors: skip entirely (e.g. concat_mem when not concat)
                if exposed.origin_tensor._element_count == 0:
                    exposed._serialized = None
                    exposed._serialized_size = 0
                    continue

                # Alias targets share buffer with source — skip serialization
                is_alias = (
                    self._alias_registry
                    and self._alias_registry.is_alias_target(name)
                )
                # Check if this tensor's send op has _skip_data (inference: BO on device)
                skip_data = any(
                    getattr(op, '_skip_data', False)
                    for op in self._ops
                    if op.tensor is not None and op.tensor.name == name
                )
                if is_alias and exposed.origin_tensor.data is None:
                    # Will use source tensor's buffer; compute size only
                    num_beats = math.ceil(
                        exposed.origin_tensor._element_count
                        / packing.elements_per_beat
                    )
                    exposed._serialized = None
                    exposed._serialized_size = num_beats * (packing.bus_width // 8)
                elif skip_data and exposed.origin_tensor.data is None:
                    # Inference: device tensor, skip serialization, compute size only
                    num_beats = math.ceil(
                        exposed.origin_tensor._element_count
                        / packing.elements_per_beat
                    )
                    exposed._serialized = None
                    exposed._serialized_size = num_beats * (packing.bus_width // 8)
                elif exposed.origin_tensor.data is None:
                    raise SerializationError(
                        f"Tensor '{name}' has no data. "
                        f"Call generate_inputs() before run()."
                    )
                else:
                    # Auto-layout: if kernel has layout_{name}(), apply it
                    serialize_data = self._apply_layout(
                        view, exposed, exposed.origin_tensor.data
                    )
                    serializer = StreamSerializer(packing)
                    exposed._serialized = serializer.serialize(serialize_data)
                    exposed._serialized_size = len(exposed._serialized)
            else:
                num_beats = math.ceil(
                    exposed.origin_tensor._element_count
                    / packing.elements_per_beat
                )
                exposed._serialized = None
                exposed._serialized_size = num_beats * (packing.bus_width // 8)

            # Multi-port split → _port_buffers
            if iface_spec.split:
                split_spec = _parse_split_spec(iface_spec.split)
                if exposed._serialized is not None:
                    splitter = MultiPortSerializer()
                    exposed._port_buffers = splitter.split_tensor(
                        exposed._serialized, split_spec
                    )
                else:
                    # Output tensor: allocate empty per-port buffers
                    n_ports = len(split_spec.ports)
                    per_port_size = exposed._serialized_size // n_ports
                    exposed._port_buffers = {
                        p.name: bytes(per_port_size)
                        for p in split_spec.ports
                    }
                exposed._port_mode = split_spec.mode
                if split_spec.interleave:
                    exposed._interleave_unit = split_spec.interleave.unit

            # Array interface → _port_buffers (when split didn't already set it)
            if iface_spec.array and not exposed._port_buffers:
                flat_names = iface_spec.array.flat_names(exposed.top_interface)
                if iface_spec.array.interleave and exposed._serialized is not None:
                    from vten.spec.models import PortDef, SplitSpec
                    pseudo_spec = SplitSpec(
                        mode="channel_interleave",
                        ports=[PortDef(name=n, base_addr=0) for n in flat_names],
                        interleave=iface_spec.array.interleave,
                    )
                    splitter = MultiPortSerializer()
                    exposed._port_buffers = splitter.split_tensor(
                        exposed._serialized, pseudo_spec
                    )
                    exposed._port_mode = "channel_interleave"
                    exposed._interleave_unit = iface_spec.array.interleave.unit
                elif iface_spec.array.interleave and exposed._serialized is None:
                    # DEV_TO_HOST: allocate empty per-port buffers
                    n_ports = len(flat_names)
                    per_port_size = exposed._serialized_size // n_ports
                    exposed._port_buffers = {
                        fn: bytes(per_port_size) for fn in flat_names
                    }
                    exposed._port_mode = "channel_interleave"
                    exposed._interleave_unit = iface_spec.array.interleave.unit
                else:
                    exposed._port_buffers = _block_split_data(
                        exposed._serialized, flat_names, exposed._serialized_size
                    )
                    exposed._port_mode = "block"

    # ── Layout helpers (delegated to runtime/layout.py) ──

    @staticmethod
    def _apply_layout(view, exposed, data):
        from vten.runtime.layout import apply_layout
        return apply_layout(view, exposed, data)

    @staticmethod
    def _apply_unlayout(view, exposed, data):
        from vten.runtime.layout import apply_unlayout
        return apply_unlayout(view, exposed, data)

    # ── Stage 3b: Dynamic Probe Mapping + Golden Serialization ──

    def _ensure_probe_mappings(
        self,
        view: FlattenedKernelView,
        internal_probe_golden: dict[tuple[str, str], object],
    ) -> None:
        """Upgrade INTERNAL → INTERNAL_PROBE for declarative probes.

        When internal_probe_golden requests a probe on a connection that
        uses Internal() without activated probes, dynamically
        upgrade the mapping type and create a ProbePoint.
        """
        existing = {
            (p.connection.source_sub, p.connection.source_name)
            for p in view.probe_points
            if p.connection is not None
        }
        for sub_name, tensor_name in internal_probe_golden:
            if (sub_name, tensor_name) in existing:
                continue
            # Find matching connection
            for conn in view.connections:
                if conn.source_sub == sub_name and conn.source_name == tensor_name:
                    # Find source interface mapping to upgrade
                    for m in view.interface_mappings:
                        if (
                            m.sub_kernel == sub_name
                            and m.sub_interface == conn.source_interface
                            and m.mapping_type == MappingType.INTERNAL
                        ):
                            m.mapping_type = MappingType.INTERNAL_PROBE
                            view.probe_points.append(
                                ProbePoint(
                                    connection=conn,
                                    interface_mapping=m,
                                )
                            )
                            break
                    break

    def _serialize_probe_golden(
        self,
        view: FlattenedKernelView,
        internal_probe_golden: dict[tuple[str, str], object] | None = None,
    ) -> None:
        """Serialize composite internal probe golden data.

        For each ProbePoint with a connection (composite internal probe),
        finds the matching golden tensor from internal_probe_golden,
        serializes it using the source interface's packing, and stores
        the result in probe.serialized_golden.
        """
        if not internal_probe_golden:
            return

        from vten.runtime.serializer import StreamSerializer

        for probe in view.probe_points:
            if probe.connection is None:
                continue

            conn = probe.connection
            key = (conn.source_sub, conn.source_name)
            golden_tensor = internal_probe_golden.get(key)
            if golden_tensor is None:
                logger.warning(
                    "No internal probe golden for %s.%s — skipping",
                    conn.source_sub, conn.source_name,
                )
                continue

            # Find the source interface packing from the sub-kernel's spec
            sub_inst = view.sub_kernels.get(conn.source_sub)
            if sub_inst is None:
                continue

            # The connection's source_interface is the sub-kernel interface name
            src_iface = sub_inst.spec.get_interface(conn.source_interface)
            if src_iface is None or src_iface.packing is None:
                logger.warning(
                    "No packing for interface %s.%s — skipping probe golden",
                    conn.source_sub, conn.source_interface,
                )
                continue

            serializer = StreamSerializer(src_iface.packing)
            serialized = serializer.serialize(golden_tensor)

            probe.golden_data = golden_tensor
            probe.serialized_golden = serialized
            probe.tensor_name = f"__probe_{conn.source_sub}_{conn.source_name}"

            logger.log(
                5, "Serialized internal probe golden: %s.%s → %d bytes",
                conn.source_sub, conn.source_name, len(serialized),
            )

    @staticmethod
    def _build_probe_buffer_map(
        view: FlattenedKernelView,
    ) -> dict[int, int]:
        """Build probe_index → golden_buffer_id mapping.

        Only includes composite internal probes (those with connection set).
        probe_index is deterministic: sequential order of probe_points with
        connections, matching the codegen order for probe BFM instantiation.
        """
        mapping: dict[int, int] = {}
        probe_index = 0
        for probe in view.probe_points:
            if probe.connection is not None and probe.golden_buffer_id is not None:
                mapping[probe_index] = probe.golden_buffer_id
                probe_index += 1
        return mapping

    @staticmethod
    def _add_probe_golden_to_view(
        view: FlattenedKernelView,
        probe_golden_tensors: dict,
    ) -> None:
        """Serialize golden tensors and add as ProbePoints to view.

        Creates ProbePoint(tensor_name=..., serialized_golden=...) entries
        in view.probe_points. These are then allocated and packed by _pack_shm()
        through the unified probe_points path.
        """
        from vten.runtime.serializer import StreamSerializer

        for tensor_name, golden_tensor in probe_golden_tensors.items():
            exposed = view.exposed_tensors.get(tensor_name)
            if exposed is None:
                continue
            iface = view.top_spec.get_interface(exposed.top_interface)
            if iface.packing is None:
                continue
            serializer = StreamSerializer(iface.packing)
            serialized = serializer.serialize(golden_tensor)
            view.probe_points.append(ProbePoint(
                tensor_name=tensor_name,
                golden_data=golden_tensor,
                serialized_golden=serialized,
            ))

    # ── Stage 4: Address Allocation ──

    def _allocate_addresses(self, view: FlattenedKernelView) -> None:
        allocators: dict[str, AddressAllocator] = {}
        for region_name, region in view.top_spec.memory_regions.items():
            allocators[region_name] = AddressAllocator(region)

        for name, exposed in view.exposed_tensors.items():
            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
            except KeyError:
                continue
            if not iface.memory_region:
                continue
            if exposed.origin_tensor._address is not None:
                continue
            if iface.memory_region not in allocators:
                continue
            addr = allocators[iface.memory_region].allocate(
                tensor_name=f"{view.name}.{name}",
                size=exposed._serialized_size,
            )
            exposed.set_address(addr)

    # ── Stage 5: auto_bind + config register Resolution ──

    def _resolve_registers(self, view: FlattenedKernelView) -> None:
        view._register_bindings = resolve_registers(view)

    # ── Stage 6b: BFM Configuration Synthesis ──

    def _synthesize_bfm_configs(
        self,
        view: FlattenedKernelView,
        commands: list[Command],
        buffer_ids: dict[str, int],
    ) -> list[BFMConfig]:
        from vten.runtime.bfm_config import synthesize_bfm_configs
        return synthesize_bfm_configs(view, commands, buffer_ids)

    @staticmethod
    def compile_multi(
        engines: list[RuntimeEngine],
        *,
        target: str = "sim",
    ) -> CompiledResult:
        """Compile multiple config groups into a single batch.

        Each engine represents one config group. Commands are merged with
        BARRIER commands inserted between groups. cmd_id and buffer_id
        are offset to maintain global uniqueness.

        Args:
            engines: List of RuntimeEngine instances, one per config group.
            target: "sim" for SIM backends, "hw" for HW backends.

        Returns:
            A single CompiledResult with merged commands and unified SHM image.
        """
        if not engines:
            raise CompilationError("compile_multi requires at least one engine")

        if len(engines) == 1:
            return engines[0].compile(target=target)

        t0 = time.perf_counter()
        logger.log(5, "compile_multi: %d config groups", len(engines))

        all_commands: list[Command] = []
        all_buffer_ids: dict[str, int] = {}
        all_bfm_configs: list[BFMConfig] = []
        all_iface_id_to_name: dict[int, str] = {}
        all_tensor_data: dict[int, bytes] = {}
        views: list[FlattenedKernelView] = []
        view_buffer_ids: list[dict[str, int]] = []

        next_cmd_id = 0
        next_buffer_id = 0
        bfm_configs_set: set[str] = set()  # dedup by interface_name

        for idx, engine in enumerate(engines):
            logger.log(5, "compile_multi: group %d/%d", idx + 1, len(engines))

            view, commands, buffer_ids, bfm_configs, iface_id_to_name = (
                engine._compile_ir(
                    cmd_id_start=next_cmd_id,
                    buffer_id_start=next_buffer_id,
                )
            )

            views.append(view)
            view_buffer_ids.append(buffer_ids)

            # Merge commands
            all_commands.extend(commands)

            # Prefix buffer_ids with config index to avoid name collision
            for name, bid in buffer_ids.items():
                all_buffer_ids[f"cfg{idx}:{name}"] = bid
                # Also keep unprefixed for single-config backward compat
                if idx == 0:
                    all_buffer_ids[name] = bid

            # Update next offsets
            if commands:
                next_cmd_id = max(c.cmd_id for c in commands) + 1
            if buffer_ids:
                next_buffer_id = max(buffer_ids.values()) + 1

            # Merge BFM configs (dedup by interface_name)
            for cfg in bfm_configs:
                if cfg.interface_name not in bfm_configs_set:
                    bfm_configs_set.add(cfg.interface_name)
                    all_bfm_configs.append(cfg)

            # Merge iface_id_to_name
            all_iface_id_to_name.update(iface_id_to_name)

            # Collect tensor data
            td = RuntimeEngine._collect_tensor_data(view, buffer_ids)
            all_tensor_data.update(td)

            # Insert BARRIER between config groups (not after last)
            if idx < len(engines) - 1:
                barrier_cmd = Command(
                    op=OpCode.BARRIER,
                    cmd_id=next_cmd_id,
                    sync=True,
                )
                all_commands.append(barrier_cmd)
                next_cmd_id += 1

        # Stage 7: SHM packing with merged data
        if target == "sim":
            logger.log(5, "Stage 7: multi-config SHM packing")
            shm_image = pack_shm_multi(views, view_buffer_ids, all_commands)
            logger.log(5, "SHM image: %d bytes", len(shm_image))
        else:
            shm_image = b""

        elapsed = time.perf_counter() - t0
        logger.debug("compile_multi complete: %d commands, %d buffers, %.1fms",
                     len(all_commands), len(all_buffer_ids), elapsed * 1000)

        # Collect all ops from all engines
        all_ops = []
        for eng in engines:
            all_ops.extend(eng._ops)

        return CompiledResult(
            commands=all_commands,
            shm_image=shm_image,
            bfm_configs=all_bfm_configs,
            buffer_ids=all_buffer_ids,
            flattened_view=views[0],  # Primary view for output tensor reading
            ops=all_ops,
            probe_reports=views[0].probe_points,
            tensor_data=all_tensor_data,
            iface_id_to_name=all_iface_id_to_name,
            views=views,  # all views for multi-config verify
        )

