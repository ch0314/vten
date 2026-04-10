"""RuntimeEngine — 8-stage compile pipeline orchestrator.

Spec reference: 02_runtime_engine.md §4
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from vten.errors import (
    CompilationError,
    ConnectionShapeMismatchError,
    SerializationError,
    ShapeMismatchError,
)
from vten.spec.models import (
    Direction,
    MappingType,
    OpCode,
)
from vten.runtime.address import AddressAllocator
from vten.runtime.binder import resolve_registers
from vten.runtime.kernel_view import (
    ExposedTensor,
    FlattenedKernelView,
    KernelInstance,
    ProbePoint,
)
from vten.runtime.flatten import (
    flatten_composite,
    is_composite,
    wrap_unit_as_flat,
)
from vten.runtime.ir import BFMConfig, Command, IRLowering
from vten.runtime.resolver import ParameterResolver
from vten.runtime.serializer import (
    MultiPortSerializer,
    StreamSerializer,
    block_split_data,
    parse_split_spec,
)

if TYPE_CHECKING:
    from vten.dsl.operations import Operation
    from vten.runtime.context import AliasRegistry


# ── CompiledResult ──


@dataclass
class CompiledResult:
    commands: list[Command]
    bfm_configs: list[BFMConfig]
    buffer_ids: dict[str, int]
    flattened_view: FlattenedKernelView
    ops: list[Operation] = field(default_factory=list)  # original DSL operations
    probe_reports: list[ProbePoint] = field(default_factory=list)
    tensor_data: dict[int, bytes] = field(default_factory=dict)
    iface_id_to_name: dict[int, str] = field(default_factory=dict)
    views: list[FlattenedKernelView] | None = None  # multi-config: all views
    view_buffer_ids: list[dict[str, int]] | None = None  # multi-config: per-view buffer_ids
    probe_buffer_map: dict[int, int] = field(default_factory=dict)  # probe_index → golden_buffer_id
    prebound_buffers: dict[int, object] = field(default_factory=dict)  # buffer_id → xrt.bo (inference)
    mode: str = "verification"  # "verification" or "inference"

    @property
    def shm_image(self) -> bytes:
        """Pack SHM image on demand (for tests / backward compat)."""
        if self.flattened_view is None:
            return b""
        from vten.backend.sim.shm_packer import pack_shm, pack_shm_multi

        if self.views and len(self.views) > 1:
            view_bids = self.view_buffer_ids or []
            return pack_shm_multi(self.views, view_bids, self.commands)
        img, _ = pack_shm(
            self.flattened_view, self.commands,
            self.buffer_ids, self.ops,
        )
        return img


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
        project_dir: Path | None = None,
    ) -> None:
        self._kernels = kernels
        self._ops = ops
        self._project_params = project_params
        self._alias_registry = alias_registry
        self._quiet = quiet
        self._project_dir = project_dir

    # ── Internal: Stages 0–9 (IR generation, no SHM packing) ──

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
        """Run Stages 0–9 and return intermediate results.

        Returns:
            (view, commands, buffer_ids, bfm_configs, iface_id_to_name)
        """
        kernel = self._get_primary_kernel()

        # Stage 0: Flatten or wrap
        logger.log(5, "Stage 0: flatten/wrap")
        if is_composite(kernel):
            view = flatten_composite(kernel, self._project_params,
                                       project_dir=self._project_dir)
        else:
            view = wrap_unit_as_flat(kernel)

        # Stage 1: Parameter resolution (re-validate)
        logger.log(5, "Stage 1: parameter resolution")
        self._resolve_parameters(view)

        # Stage 2: Shape resolution & validation
        logger.log(5, "Stage 2: shape resolution")
        self._resolve_shapes(view)

        # Stage 3: Direction refinement from operations
        logger.log(5, "Stage 3: direction refinement")
        self._refine_directions_from_ops(view)

        # Stage 4: Tensor serialization
        logger.log(5, "Stage 4: tensor serialization")
        self._serialize_tensors(view)

        # Stage 5: Probe golden serialization
        logger.log(5, "Stage 5: probe golden serialization")
        self._serialize_probe_golden(view)

        # Stage 6: Address allocation
        logger.log(5, "Stage 6: address allocation")
        self._allocate_addresses(view)

        # Stage 7: auto_bind resolution
        logger.log(5, "Stage 7: auto_bind resolution")
        self._resolve_registers(view)

        # Stage 8: IR lowering
        logger.log(5, "Stage 8: IR lowering")
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

        # Stage 9: BFM configuration synthesis
        logger.log(5, "Stage 9: BFM config synthesis")
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
        probe_golden_tensors: dict | None = None,
        internal_probe_golden: dict | None = None,
    ) -> CompiledResult:
        """Run the compile pipeline (Stages 0–9).

        Produces backend-agnostic IR commands + tensor data.
        SHM packing is handled by SimBackend.execute().

        Args:
            probe_golden_tensors: tensor_name → torch.Tensor golden data for probe.
            internal_probe_golden: (sub_name, tensor_name) → torch.Tensor for
                composite internal probe golden data.
        """
        t0 = time.perf_counter()
        logger.log(5, "compile pipeline starting (ops=%d)", len(self._ops))

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

        # Build probe_index → golden_buffer_id mapping
        probe_buffer_map = self._build_probe_buffer_map(view)

        tensor_data = self._collect_tensor_data(view, buffer_ids)

        elapsed = time.perf_counter() - t0
        logger.debug("compile complete: %d commands, %d buffers, %.1fms",
                     len(commands), len(buffer_ids), elapsed * 1000)

        return CompiledResult(
            commands=commands,
            bfm_configs=bfm_configs,
            buffer_ids=buffer_ids,
            flattened_view=view,
            ops=list(self._ops),
            probe_reports=view.probe_points,
            tensor_data=tensor_data,
            iface_id_to_name=iface_id_to_name,
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

    # Stage 0 logic (flatten/wrap/validate/direction) is in vten.runtime.flatten

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

    # ── Stage 3: Direction Refinement from Operations ──

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

    # ── Stage 4: Tensor Serialization ──

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
                # Check if this tensor's send op has _device_resident (inference: BO on device)
                skip_data = any(
                    getattr(op, '_device_resident', False)
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
                split_spec = parse_split_spec(iface_spec.split)
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
                    exposed._port_buffers = block_split_data(
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

    # ── Stage 5: Dynamic Probe Mapping + Golden Serialization ──

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

    # ── Stage 6: Address Allocation ──

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

    # ── Stage 7: auto_bind + config register Resolution ──

    def _resolve_registers(self, view: FlattenedKernelView) -> None:
        view._register_bindings = resolve_registers(view)

    # ── Stage 9: BFM Configuration Synthesis ──

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
    ) -> CompiledResult:
        """Compile multiple config groups into a single batch.

        Each engine represents one config group. Commands are merged with
        BARRIER commands inserted between groups. cmd_id and buffer_id
        are offset to maintain global uniqueness.

        Returns:
            A single CompiledResult with merged commands.
        """
        if not engines:
            raise CompilationError("compile_multi requires at least one engine")

        if len(engines) == 1:
            return engines[0].compile()

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

        elapsed = time.perf_counter() - t0
        logger.debug("compile_multi complete: %d commands, %d buffers, %.1fms",
                     len(all_commands), len(all_buffer_ids), elapsed * 1000)

        # Collect all ops from all engines
        all_ops = []
        for eng in engines:
            all_ops.extend(eng._ops)

        return CompiledResult(
            commands=all_commands,
            bfm_configs=all_bfm_configs,
            buffer_ids=all_buffer_ids,
            flattened_view=views[0],  # Primary view for output tensor reading
            ops=all_ops,
            probe_reports=views[0].probe_points,
            tensor_data=all_tensor_data,
            iface_id_to_name=all_iface_id_to_name,
            views=views,  # all views for multi-config verify
            view_buffer_ids=view_buffer_ids,
        )

