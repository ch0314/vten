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
from vten.kernel.composite import Internal
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
    resolve_auto_binds,
    resolve_config_registers,
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
from vten.runtime.shm import (
    BUF_DESC_SIZE,
    CACHE_LINE,
    CMD_SLOT_SIZE,
    CONTROL_SIZE,
    DIRECTION_ENCODING,
    STATS_SLOT_SIZE,
    SHMBufferAllocator,
    calculate_shm_size,
    pack_buffer_descriptor,
    pack_command_slot,
    pack_control_header,
    pack_stats_entry,
)

if TYPE_CHECKING:
    from vten.dsl.operations import Operation
    from vten.runtime.context import AliasRegistry


# ── Helpers ──


def _pack_shm_common(
    shm_alloc: SHMBufferAllocator,
    commands: list[Command],
    *,
    flags: int = 0,
) -> tuple[bytearray, SHMLayout, int]:
    """Pack SHM header, command slots, stats, and buffer descriptors.

    Shared by _pack_shm (single-config) and _pack_shm_multi (multi-config).
    Callers are responsible for allocating data buffers into *shm_alloc*
    before calling, and for copying tensor data into the returned image
    after this function returns.

    Returns:
        (image, layout, data_region_offset) — mutable image, layout metadata,
        and the byte offset where the data region starts.
    """
    from vten.runtime.shm import FLAG_STATS_ENABLED
    from vten.spec.models import CommandStatus

    num_commands = len(commands)
    num_buffers = len(shm_alloc.descriptors)

    total = calculate_shm_size(
        num_commands=num_commands,
        num_buffers=num_buffers,
        buffer_sizes=[d.size for d in shm_alloc.descriptors],
    )

    image = bytearray(total)

    # Region offsets
    cmd_offset = CONTROL_SIZE
    stats_offset = cmd_offset + CMD_SLOT_SIZE * num_commands
    bufdesc_offset = stats_offset + STATS_SLOT_SIZE * num_commands
    data_region_raw = bufdesc_offset + BUF_DESC_SIZE * num_buffers
    data_region_offset = (data_region_raw + CACHE_LINE - 1) & ~(CACHE_LINE - 1)

    layout = SHMLayout(
        cmd_offset=cmd_offset,
        stats_offset=stats_offset,
        bufdesc_offset=bufdesc_offset,
        data_region_offset=data_region_offset,
        num_commands=num_commands,
        num_buffers=num_buffers,
        total_size=total,
    )

    # Pack control header
    shm_flags = flags | FLAG_STATS_ENABLED
    pack_control_header(
        image,
        num_commands=num_commands,
        num_buffers=num_buffers,
        cmd_region_offset=cmd_offset,
        stats_region_offset=stats_offset,
        buf_desc_offset=bufdesc_offset,
        data_region_offset=data_region_offset,
        total_shm_size=total,
        flags=shm_flags,
    )

    # Pack command slots
    for i, cmd in enumerate(commands):
        pack_command_slot(image, cmd_offset + i * CMD_SLOT_SIZE, cmd)

    # Pack stats entries: LOAD commands pre-marked as COMMITTED
    for cmd in commands:
        if cmd.op == OpCode.LOAD:
            pack_stats_entry(
                image,
                stats_offset + cmd.cmd_id * STATS_SLOT_SIZE,
                status=CommandStatus.COMMITTED.value,
            )

    # Pack buffer descriptors
    for i, desc in enumerate(shm_alloc.descriptors):
        pack_buffer_descriptor(
            image, bufdesc_offset + i * BUF_DESC_SIZE, desc
        )

    return image, layout, data_region_offset


def _parse_split_spec(raw):
    """Parse a raw dict or SplitSpec into a SplitSpec dataclass."""
    from vten.spec.models import InterleaveSpec, PortDef, SplitSpec

    if isinstance(raw, SplitSpec):
        return raw
    ports = [
        PortDef(name=p["name"], base_addr=p.get("base_addr", 0))
        for p in raw.get("ports", [])
    ]
    interleave = None
    if "interleave" in raw:
        interleave = InterleaveSpec(unit=raw["interleave"]["unit"])
    return SplitSpec(mode=raw["mode"], ports=ports, interleave=interleave)


def _block_split_data(
    serialized: bytes | None,
    flat_names: list[str],
    serialized_size: int,
) -> dict[str, bytes]:
    """Block-split serialized data (or allocate empty) across port names."""
    n = len(flat_names)
    if serialized is not None:
        data = serialized
        chunk_size = len(data) // n
        remainder = len(data) % n
        result = {}
        offset = 0
        for i, fname in enumerate(flat_names):
            sz = chunk_size + (1 if i < remainder else 0)
            result[fname] = data[offset : offset + sz]
            offset += sz
        return result
    else:
        per_elem_size = serialized_size // n
        return {fname: bytes(per_elem_size) for fname in flat_names}


# ── CompiledResult ──


@dataclass
class SHMLayout:
    """SHM region offsets and counts for in-place batch updates."""

    cmd_offset: int
    stats_offset: int
    bufdesc_offset: int
    data_region_offset: int
    num_commands: int
    num_buffers: int
    total_size: int


@dataclass
class CompiledResult:
    commands: list[Command]
    shm_image: bytes
    bfm_configs: list[BFMConfig]
    buffer_ids: dict[str, int]
    flattened_view: FlattenedKernelView
    probe_reports: list[ProbePoint] = field(default_factory=list)
    tensor_data: dict[int, bytes] = field(default_factory=dict)
    iface_id_to_name: dict[int, str] = field(default_factory=dict)
    shm_layout: SHMLayout | None = None
    views: list[FlattenedKernelView] | None = None  # multi-config: all views
    probe_buffer_map: dict[int, int] = field(default_factory=dict)  # probe_index → golden_buffer_id


# ── RuntimeEngine ──


class RuntimeEngine:
    """8-stage compile pipeline orchestrator."""

    def __init__(
        self,
        kernels: dict[str, KernelInstance],
        ops: list[Operation],
        project_params: dict,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self._kernels = kernels
        self._ops = ops
        self._project_params = project_params
        self._alias_registry = alias_registry

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
        logger.debug("Stage 0: flatten/wrap")
        if self._is_composite(kernel):
            view = self._flatten_composite(kernel)
        else:
            view = self._wrap_unit_as_flat(kernel)

        # Stage 1: Parameter resolution (re-validate)
        logger.debug("Stage 1: parameter resolution")
        self._resolve_parameters(view)

        # Stage 2: Shape resolution & validation
        logger.debug("Stage 2: shape resolution")
        self._resolve_shapes(view)

        # Stage 2b: Refine direction from operations
        logger.debug("Stage 2b: direction refinement")
        self._refine_directions_from_ops(view)

        # Stage 3: Tensor serialization
        logger.debug("Stage 3: tensor serialization")
        self._serialize_tensors(view)

        # Stage 3b: Probe golden serialization
        logger.debug("Stage 3b: probe golden serialization")
        self._serialize_probe_golden(view)

        # Stage 4: Address allocation
        logger.debug("Stage 4: address allocation")
        self._allocate_addresses(view)

        # Stage 5: auto_bind resolution
        logger.debug("Stage 5: auto_bind resolution")
        self._resolve_auto_binds(view)

        # Stage 6: IR lowering
        logger.debug("Stage 6: IR lowering")
        lowering = IRLowering(view, self._alias_registry)
        commands, buffer_ids = lowering.lower(
            self._ops,
            cmd_id_start=cmd_id_start,
            buffer_id_start=buffer_id_start,
        )

        iface_id_to_name: dict[int, str] = {
            v: k for k, v in lowering._iface_id_map.items()
        }

        if logger.isEnabledFor(logging.DEBUG):
            self._log_ir_commands(commands, iface_id_to_name, buffer_ids)

        # Stage 6b: BFM configuration synthesis
        logger.debug("Stage 6b: BFM config synthesis")
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
        logger.debug("compile pipeline starting (target=%s, ops=%d)", target, len(self._ops))

        view, commands, buffer_ids, bfm_configs, iface_id_to_name = (
            self._compile_ir()
        )

        # Populate view.probe_points for single-kernel probe golden tensors
        if probe_golden_tensors:
            self._add_probe_golden_to_view(view, probe_golden_tensors)

        # Serialize composite internal probe golden data
        if internal_probe_golden:
            self._serialize_probe_golden(view, internal_probe_golden)

        # Stage 7: SHM packing (SIM path only)
        probe_buffer_map: dict[int, int] = {}
        if target == "sim":
            logger.debug("Stage 7: SHM packing")
            shm_image = self._pack_shm(view, commands, buffer_ids, flags=flags)
            logger.debug("SHM image: %d bytes", len(shm_image))
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
            probe_reports=view.probe_points,
            tensor_data=tensor_data,
            iface_id_to_name=iface_id_to_name,
            shm_layout=self._last_shm_layout,
            probe_buffer_map=probe_buffer_map,
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
        return hasattr(kernel.kernel_class, "_sub_kernel_bindings") and bool(
            kernel.kernel_class._sub_kernel_bindings
        )

    def _flatten_composite(self, kernel: KernelInstance) -> FlattenedKernelView:
        """Flatten a CompositeKernel into FlattenedKernelView.

        Spec reference: 02_runtime_engine.md §5 (5-phase algorithm).
        """
        from vten.kernel.composite import CompositeKernel, Connect, Internal
        from vten.spec.parser import load_kernel_spec

        composite_instance = kernel.kernel_class_instance
        top_spec = kernel.spec

        # ── Phase A: Sub-kernel instantiation ──
        # Reuse sub-kernel instances created during initialize() if available
        existing_subs = kernel._sub_kernel_instances
        sub_kernels: dict[str, KernelInstance] = {}
        bindings_map: dict[str, object] = {}  # name → SubKernelBinding

        for bind_name, binding in composite_instance.bindings():
            if existing_subs and bind_name in existing_subs:
                sub_ki = existing_subs[bind_name]
                # Ensure spec is fully loaded (initialize may have used fallback)
                sub_spec_path = binding.kernel_class.spec
                if sub_spec_path and not sub_ki.spec.interfaces:
                    sub_ki.spec = load_kernel_spec(sub_spec_path)
            else:
                sub_spec_path = binding.kernel_class.spec
                sub_spec = load_kernel_spec(sub_spec_path)
                sub_ki = KernelInstance(
                    name=bind_name,
                    spec=sub_spec,
                    kernel_class=binding.kernel_class,
                    runtime_params=binding.params or {},
                )
                sub_ki.initialize(self._project_params)
            sub_kernels[bind_name] = sub_ki
            bindings_map[bind_name] = binding

        # Synthesize top-level spec for composite kernels if missing
        if not top_spec.interfaces:
            # Check class-level cache first
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

        # ── Phase B: Interface mapping construction ──
        mappings: list[InterfaceMapping] = []

        for bind_name, binding in composite_instance.bindings():
            sub_spec = sub_kernels[bind_name].spec
            for sub_iface_name in sub_spec.interface_names():
                if sub_iface_name not in binding.interface_map:
                    raise ValidationError(
                        f"CompositeKernel '{kernel.name}': sub-kernel "
                        f"'{bind_name}' interface '{sub_iface_name}' "
                        f"has no mapping in interface_map."
                    )
                mapping_value = binding.interface_map[sub_iface_name]
                mappings.append(
                    self._parse_mapping(
                        bind_name, sub_iface_name, mapping_value, top_spec
                    )
                )

        # ── Phase C: Exposed tensor collection ──
        exposed: dict[str, ExposedTensor] = {}

        for tensor_attr, tensor_def in composite_instance.exposed_tensor_defs():
            origin_sub = tensor_def.origin_sub_kernel
            origin_name = tensor_def.origin_name

            if origin_sub not in sub_kernels:
                raise ValidationError(
                    f"ExposedTensor '{tensor_attr}' references sub-kernel "
                    f"'{origin_sub}' which does not exist."
                )

            origin_tensor = sub_kernels[origin_sub].get_tensor(origin_name)
            direction = self._infer_direction_composite(
                origin_sub, origin_tensor, mappings, sub_kernels[origin_sub].spec
            )

            exposed[tensor_attr] = ExposedTensor(
                name=tensor_attr,
                origin_path=f"{origin_sub}.{origin_name}",
                origin_tensor=origin_tensor,
                top_interface=tensor_def.top_interface,
                direction=direction,
            )

        # ── Phase D: Probe point collection ──
        # Must iterate connections in list order to match codegen probe_index
        # assignment (build/composite.py extract_probe_bfm_info).
        probe_points: list[ProbePoint] = []
        connections = composite_instance._connections or []

        # Build set of probed (sub_name, sub_interface) from INTERNAL_PROBE mappings
        probed_ifaces: set[tuple[str, str]] = set()
        probe_mapping_by_key: dict[tuple[str, str], InterfaceMapping] = {}
        for m in mappings:
            if m.mapping_type == MappingType.INTERNAL_PROBE:
                key = (m.sub_kernel, m.sub_interface)
                probed_ifaces.add(key)
                probe_mapping_by_key[key] = m

        for conn in connections:
            # Check if either end of this connection is probed
            src_key = (conn.source_sub, conn.source_interface)
            dst_iface = self._find_dest_interface_name(conn, mappings)
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

        return FlattenedKernelView(
            name=kernel.name,
            top_spec=top_spec,
            sub_kernels=sub_kernels,
            interface_mappings=mappings,
            exposed_tensors=exposed,
            probe_points=probe_points,
            connections=connections,
        )

    def _parse_mapping(
        self,
        sub_kernel_name: str,
        sub_iface_name: str,
        mapping_value: object,
        top_spec: KernelSpec,
    ) -> InterfaceMapping:
        """Parse a single interface_map entry into InterfaceMapping.

        Handles three mapping forms:
          - Internal() / Internal(probe=True)
          - "top_interface_name" (string → EXTERNAL)
          - ("top_interface_name", "bank_name") (tuple → EXTERNAL_BANK)
        """
        if isinstance(mapping_value, Internal):
            mtype = (
                MappingType.INTERNAL_PROBE
                if mapping_value.probe
                else MappingType.INTERNAL
            )
            return InterfaceMapping(
                sub_kernel=sub_kernel_name,
                sub_interface=sub_iface_name,
                mapping_type=mtype,
                top_interface=None,
                bank_name=None,
                bank_offset=0,
            )

        if isinstance(mapping_value, str):
            return InterfaceMapping(
                sub_kernel=sub_kernel_name,
                sub_interface=sub_iface_name,
                mapping_type=MappingType.EXTERNAL,
                top_interface=mapping_value,
                bank_name=None,
                bank_offset=0,
            )

        if isinstance(mapping_value, tuple) and len(mapping_value) == 2:
            top_iface, bank_name = mapping_value
            bank_offset = top_spec.get_bank_offset(top_iface, bank_name)
            return InterfaceMapping(
                sub_kernel=sub_kernel_name,
                sub_interface=sub_iface_name,
                mapping_type=MappingType.EXTERNAL_BANK,
                top_interface=top_iface,
                bank_name=bank_name,
                bank_offset=bank_offset,
            )

        raise ValidationError(
            f"Invalid interface_map value for '{sub_kernel_name}."
            f"{sub_iface_name}': {mapping_value!r}. "
            f"Expected Internal(), string, or (string, string) tuple."
        )

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

    def _find_connection_for_interface(
        self,
        connections: list,
        sub_kernel_name: str,
        sub_interface_name: str,
    ) -> object | None:
        """Find a Connect involving the given sub-kernel interface."""
        for conn in connections:
            if conn.source_sub == sub_kernel_name and conn.source_interface == sub_interface_name:
                return conn
            if conn.dest_sub == sub_kernel_name:
                from vten.kernel.tensor import Tensor

                dest_tensor = getattr(conn._dest_proxy.kernel_class, conn.dest_name, None)
                if isinstance(dest_tensor, Tensor) and dest_tensor.interface == sub_interface_name:
                    return conn
        return None

    @staticmethod
    def _find_dest_interface_name(conn, mappings: list) -> str | None:
        """Find the destination interface name from a Connect object."""
        from vten.kernel.tensor import Tensor

        dest_tensor = getattr(conn._dest_proxy.kernel_class, conn.dest_name, None)
        if isinstance(dest_tensor, Tensor):
            return dest_tensor.interface
        return None

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

        Even for internal (RTL wire) connections, dtype must match
        unless an explicit *transform* is provided on the Connect.
        """
        for conn in connections:
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

        h2d_ops = {
            OpKind.SEND_TENSOR, OpKind.PUSH_TENSOR, OpKind.LOAD_TENSOR,
        }
        d2h_ops = {
            OpKind.RECV_TENSOR, OpKind.PULL_TENSOR, OpKind.STORE_TENSOR,
        }

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
                # Alias targets share buffer with source — skip serialization
                is_alias = (
                    self._alias_registry
                    and self._alias_registry.is_alias_target(name)
                )
                if is_alias and exposed.origin_tensor.data is None:
                    # Will use source tensor's buffer; compute size only
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
                    serializer = StreamSerializer(packing)
                    exposed._serialized = serializer.serialize(
                        exposed.origin_tensor.data
                    )
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

    # ── Stage 3b: Probe Golden Serialization ──

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

            logger.debug(
                "Serialized internal probe golden: %s.%s → %d bytes",
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

    def _resolve_auto_binds(self, view: FlattenedKernelView) -> None:
        auto_bindings = resolve_auto_binds(view)
        config_bindings = resolve_config_registers(view)
        view._register_bindings = auto_bindings + config_bindings

    # ── Stage 6b: BFM Configuration Synthesis ──

    def _synthesize_bfm_configs(
        self,
        view: FlattenedKernelView,
        commands: list[Command],
        buffer_ids: dict[str, int],
    ) -> list[BFMConfig]:
        bfm_configs: dict[str, BFMConfig] = {}

        for top_iface_name in view.external_interfaces():
            iface_spec = view.top_spec.get_interface(top_iface_name)

            if iface_spec.protocol in (Protocol.AXI4, Protocol.AXI4S):
                address_ranges: list[tuple[int, int, int]] = []
                for exposed in view.tensors_for_interface(top_iface_name):
                    if exposed.address is not None:
                        address_ranges.append(
                            (
                                exposed.address,
                                exposed._serialized_size,
                                buffer_ids[exposed.name],
                            )
                        )

                if iface_spec.array:
                    # Expand array interface into N individual BFMs
                    for flat_name in iface_spec.array.flat_names(top_iface_name):
                        bfm_configs[flat_name] = BFMConfig(
                            interface_name=flat_name,
                            protocol=iface_spec.protocol,
                            data_width=iface_spec.data_width or 256,
                            role="slave" if iface_spec.protocol == Protocol.AXI4 else "master",
                            address_ranges=sorted(address_ranges),
                        )
                else:
                    bfm_configs[top_iface_name] = BFMConfig(
                        interface_name=top_iface_name,
                        protocol=iface_spec.protocol,
                        data_width=iface_spec.data_width or 256,
                        role="slave" if iface_spec.protocol == Protocol.AXI4 else "master",
                        address_ranges=sorted(address_ranges),
                    )
            elif iface_spec.protocol == Protocol.AXI4L:
                bfm_configs[top_iface_name] = BFMConfig(
                    interface_name=top_iface_name,
                    protocol=Protocol.AXI4L,
                    data_width=32,
                    role="master",
                )

        return list(bfm_configs.values())

    # ── Stage 7: SHM Packing ──

    _last_shm_layout: SHMLayout | None = None

    def _pack_shm(
        self,
        view: FlattenedKernelView,
        commands: list[Command],
        buffer_ids: dict[str, int],
        flags: int = 0,
    ) -> bytes:
        shm_alloc = SHMBufferAllocator()

        # Allocate data buffers
        allocated_buffer_ids: set[int] = set()

        # Collect chunk info from ops to detect chunked tensors
        chunk_tensors: dict[str, int | list[int]] = {}
        for op in self._ops:
            if op.chunk_total is not None and op.tensor is not None:
                chunk_tensors[op.tensor.name] = op.chunks_spec

        for name, exposed in view.exposed_tensors.items():
            direction = DIRECTION_ENCODING.get(exposed.direction, 0)

            if name in chunk_tensors:
                # Chunked tensor: allocate per-chunk buffers
                chunks_spec = chunk_tensors[name]
                n_chunks = (
                    len(chunks_spec) if isinstance(chunks_spec, list)
                    else chunks_spec
                )
                for ci in range(n_chunks):
                    if isinstance(chunks_spec, list):
                        total_elems = sum(chunks_spec)
                        chunk_size = (
                            exposed._serialized_size * chunks_spec[ci]
                            // total_elems
                        )
                    else:
                        chunk_size = exposed._serialized_size // n_chunks

                    if exposed._port_buffers:
                        flat_names = list(
                            exposed._port_buffers.keys()
                        )
                        n_elems = len(flat_names)
                        per_elem_size = chunk_size // n_elems
                        for fname in flat_names:
                            bid = buffer_ids[
                                f"{name}:chunk_{ci}:{fname}"
                            ]
                            if bid not in allocated_buffer_ids:
                                allocated_buffer_ids.add(bid)
                                shm_alloc.allocate(
                                    bid, per_elem_size, direction,
                                )
                    else:
                        bid = buffer_ids[f"{name}:chunk_{ci}"]
                        if bid not in allocated_buffer_ids:
                            allocated_buffer_ids.add(bid)
                            shm_alloc.allocate(bid, chunk_size, direction)
            elif exposed._port_buffers:
                # Array tensor: one buffer per flat element
                for flat_name, chunk in exposed._port_buffers.items():
                    bid = buffer_ids[f"{name}:{flat_name}"]
                    if bid not in allocated_buffer_ids:
                        allocated_buffer_ids.add(bid)
                        shm_alloc.allocate(bid, len(chunk), direction)
            else:
                bid = buffer_ids[name]
                if bid not in allocated_buffer_ids:
                    allocated_buffer_ids.add(bid)
                    shm_alloc.allocate(bid, exposed._serialized_size, direction)

        # Probe golden buffers (unified: composite + single kernel)
        # For split tensors, create per-port golden buffers matching the data split.
        next_buffer_id = max(buffer_ids.values(), default=-1) + 1
        # Track per-port golden: cmd.buffer_id → (golden_buffer_id, golden_bytes)
        probe_port_golden: dict[int, tuple[int, bytes]] = {}
        for probe in view.probe_points:
            if probe.serialized_golden is None:
                continue
            # Check if tensor is split into ports
            exposed = (
                view.exposed_tensors.get(probe.tensor_name)
                if probe.tensor_name else None
            )
            if exposed and exposed._port_buffers:
                # Split golden matching the tensor data split order
                golden_bytes = probe.serialized_golden
                offset = 0
                for port_name, port_data in exposed._port_buffers.items():
                    port_size = len(port_data)
                    golden_chunk = golden_bytes[offset:offset + port_size]
                    shm_alloc.allocate(
                        next_buffer_id, port_size, 0, flags=0x01
                    )
                    port_bid = buffer_ids.get(
                        f"{probe.tensor_name}:{port_name}"
                    )
                    if port_bid is not None:
                        probe_port_golden[port_bid] = (
                            next_buffer_id, golden_chunk
                        )
                    next_buffer_id += 1
                    offset += port_size
            else:
                # Non-split: single golden buffer
                shm_alloc.allocate(
                    next_buffer_id, len(probe.serialized_golden), 0,
                    flags=0x01,
                )
                probe.golden_buffer_id = next_buffer_id
                next_buffer_id += 1

        # Assign cmd.golden_buf for probe PULL commands
        # Non-split: map by base tensor name
        probe_tensor_bids = {
            p.tensor_name: p.golden_buffer_id
            for p in view.probe_points
            if p.tensor_name and p.golden_buffer_id is not None
        }
        for cmd in commands:
            if cmd.probe and cmd.op == OpCode.PULL:
                # Split tensor: direct buffer_id → golden_buffer_id map
                if cmd.buffer_id in probe_port_golden:
                    cmd.golden_buf = probe_port_golden[cmd.buffer_id][0]
                elif probe_tensor_bids:
                    # Non-split: reverse map buffer_id → tensor name
                    for tname, bid in buffer_ids.items():
                        base = tname.split(":")[0] if ":" in tname else tname
                        if bid == cmd.buffer_id and base in probe_tensor_bids:
                            cmd.golden_buf = probe_tensor_bids[base]
                            break

        image, layout, data_region_offset = _pack_shm_common(
            shm_alloc, commands, flags=flags,
        )

        # Store layout metadata for session-based batch updates
        self._last_shm_layout = layout

        # Copy input tensor data
        for name, exposed in view.exposed_tensors.items():
            if exposed._port_buffers:
                # Array tensor: copy per-element data chunks
                for flat_name, chunk in exposed._port_buffers.items():
                    if not chunk or all(b == 0 for b in chunk):
                        continue  # Skip zero-filled output placeholders
                    bid = buffer_ids[f"{name}:{flat_name}"]
                    try:
                        desc = shm_alloc.get_descriptor(bid)
                    except KeyError:
                        continue
                    start = data_region_offset + desc.data_offset
                    image[start : start + len(chunk)] = chunk
            elif exposed._serialized is not None:
                bid = buffer_ids[name]
                try:
                    desc = shm_alloc.get_descriptor(bid)
                except KeyError:
                    continue
                start = data_region_offset + desc.data_offset
                image[start : start + len(exposed._serialized)] = (
                    exposed._serialized
                )

        # Copy probe golden data (unified + per-port split)
        for probe in view.probe_points:
            if probe.serialized_golden is not None and probe.golden_buffer_id is not None:
                desc = shm_alloc.get_descriptor(probe.golden_buffer_id)
                start = data_region_offset + desc.data_offset
                image[start : start + len(probe.serialized_golden)] = (
                    probe.serialized_golden
                )
        # Copy per-port split golden data
        for _port_bid, (golden_bid, golden_chunk) in probe_port_golden.items():
            desc = shm_alloc.get_descriptor(golden_bid)
            start = data_region_offset + desc.data_offset
            image[start : start + len(golden_chunk)] = golden_chunk

        return bytes(image)

    # ── Multi-config compile ──

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
        logger.debug("compile_multi: %d config groups", len(engines))

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
            logger.debug("compile_multi: group %d/%d", idx + 1, len(engines))

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
            logger.debug("Stage 7: multi-config SHM packing")
            shm_image = engines[0]._pack_shm_multi(
                views, view_buffer_ids, all_commands,
            )
            logger.debug("SHM image: %d bytes", len(shm_image))
        else:
            shm_image = b""

        elapsed = time.perf_counter() - t0
        logger.debug("compile_multi complete: %d commands, %d buffers, %.1fms",
                     len(all_commands), len(all_buffer_ids), elapsed * 1000)

        return CompiledResult(
            commands=all_commands,
            shm_image=shm_image,
            bfm_configs=all_bfm_configs,
            buffer_ids=all_buffer_ids,
            flattened_view=views[0],  # Primary view for output tensor reading
            probe_reports=views[0].probe_points,
            tensor_data=all_tensor_data,
            iface_id_to_name=all_iface_id_to_name,
            views=views,  # all views for multi-config verify
        )

    def _pack_shm_multi(
        self,
        views: list[FlattenedKernelView],
        view_buffer_ids: list[dict[str, int]],
        commands: list[Command],
    ) -> bytes:
        """Pack SHM image for multi-config batch with multiple views."""
        shm_alloc = SHMBufferAllocator()
        allocated_buffer_ids: set[int] = set()

        # Allocate data buffers from all views
        for view, buffer_ids in zip(views, view_buffer_ids):
            for name, exposed in view.exposed_tensors.items():
                direction = DIRECTION_ENCODING.get(exposed.direction, 0)

                if exposed._port_buffers:
                    for flat_name, chunk in exposed._port_buffers.items():
                        bid = buffer_ids[f"{name}:{flat_name}"]
                        if bid not in allocated_buffer_ids:
                            allocated_buffer_ids.add(bid)
                            shm_alloc.allocate(bid, len(chunk), direction)
                else:
                    bid = buffer_ids[name]
                    if bid not in allocated_buffer_ids:
                        allocated_buffer_ids.add(bid)
                        shm_alloc.allocate(bid, exposed._serialized_size, direction)

        image, _layout, data_region_offset = _pack_shm_common(
            shm_alloc, commands,
        )

        # Copy input tensor data from all views
        for view, buffer_ids in zip(views, view_buffer_ids):
            for name, exposed in view.exposed_tensors.items():
                if exposed._port_buffers:
                    for flat_name, chunk in exposed._port_buffers.items():
                        if not chunk or all(b == 0 for b in chunk):
                            continue
                        bid = buffer_ids[f"{name}:{flat_name}"]
                        try:
                            desc = shm_alloc.get_descriptor(bid)
                        except KeyError:
                            continue
                        start = data_region_offset + desc.data_offset
                        image[start : start + len(chunk)] = chunk
                elif exposed._serialized is not None:
                    bid = buffer_ids[name]
                    try:
                        desc = shm_alloc.get_descriptor(bid)
                    except KeyError:
                        continue
                    start = data_region_offset + desc.data_offset
                    image[start : start + len(exposed._serialized)] = (
                        exposed._serialized
                    )

        return bytes(image)
