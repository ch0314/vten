"""RuntimeEngine — 8-stage compile pipeline orchestrator.

Spec reference: 02_runtime_engine.md §4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.errors import (
    CompilationError,
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
from vten.runtime.binder import RegisterBindingEntry, resolve_auto_binds
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


# ── CompiledResult ──


@dataclass
class CompiledResult:
    commands: list[Command]
    shm_image: bytes
    bfm_configs: list[BFMConfig]
    buffer_ids: dict[str, int]
    flattened_view: FlattenedKernelView
    probe_reports: list[ProbePoint] = field(default_factory=list)
    tensor_data: dict[int, bytes] = field(default_factory=dict)


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

    def compile(self, target: str = "sim") -> CompiledResult:
        """Run the 8-stage compile pipeline.

        Args:
            target: "sim" for SIM backends (includes Stage 7 SHM packing),
                    "hw" for HW backends (skips SHM packing).
        """
        kernel = self._get_primary_kernel()

        # Stage 0: Flatten or wrap
        if self._is_composite(kernel):
            view = self._flatten_composite(kernel)
        else:
            view = self._wrap_unit_as_flat(kernel)

        # Stage 1: Parameter resolution (re-validate)
        self._resolve_parameters(view)

        # Stage 2: Shape resolution & validation
        self._resolve_shapes(view)

        # Stage 2b: Refine direction from operations (for tensors without explicit direction)
        self._refine_directions_from_ops(view)

        # Stage 3: Tensor serialization
        self._serialize_tensors(view)

        # Stage 3b: Probe golden serialization
        self._serialize_probe_golden(view)

        # Stage 4: Address allocation
        self._allocate_addresses(view)

        # Stage 5: auto_bind resolution
        self._resolve_auto_binds(view)

        # Stage 6: IR lowering
        lowering = IRLowering(view, self._alias_registry)
        commands, buffer_ids = lowering.lower(self._ops)

        # Stage 6b: BFM configuration synthesis
        bfm_configs = self._synthesize_bfm_configs(view, commands, buffer_ids)

        # Stage 7: SHM packing (SIM path only)
        if target == "sim":
            shm_image = self._pack_shm(view, commands, buffer_ids)
        else:
            shm_image = b""

        # Collect serialized tensor data (used by HW path / CommandInterpreter)
        tensor_data: dict[int, bytes] = {}
        for name, exposed in view.exposed_tensors.items():
            if exposed._serialized is not None:
                bid = buffer_ids.get(name)
                if bid is not None:
                    tensor_data[bid] = exposed._serialized

        return CompiledResult(
            commands=commands,
            shm_image=shm_image,
            bfm_configs=bfm_configs,
            buffer_ids=buffer_ids,
            flattened_view=view,
            probe_reports=view.probe_points,
            tensor_data=tensor_data,
        )

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
        sub_kernels: dict[str, KernelInstance] = {}
        bindings_map: dict[str, object] = {}  # name → SubKernelBinding

        for bind_name, binding in composite_instance.bindings():
            # Load sub-kernel spec
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
        probe_points: list[ProbePoint] = []
        connections = composite_instance._connections or []

        for m in mappings:
            if m.mapping_type == MappingType.INTERNAL_PROBE:
                conn = self._find_connection_for_interface(
                    connections, m.sub_kernel, m.sub_interface
                )
                if conn is not None:
                    probe_points.append(
                        ProbePoint(connection=conn, interface_mapping=m)
                    )

        # ── Phase E: Validation ──
        self._validate_flattened(mappings, exposed, connections, top_spec)

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

    def _validate_flattened(
        self,
        mappings: list[InterfaceMapping],
        exposed: dict[str, ExposedTensor],
        connections: list,
        top_spec: KernelSpec,
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
        for conn in view.connections:
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

            # Multi-port split
            if iface_spec.split and exposed._serialized is not None:
                from vten.spec.models import SplitSpec

                split_spec = iface_spec.split
                if isinstance(split_spec, dict):
                    # Parse raw dict to SplitSpec
                    from vten.spec.models import InterleaveSpec, PortDef

                    ports = [
                        PortDef(name=p["name"], base_addr=p.get("base_addr", 0))
                        for p in split_spec.get("ports", [])
                    ]
                    interleave = None
                    if "interleave" in split_spec:
                        interleave = InterleaveSpec(
                            unit=split_spec["interleave"]["unit"]
                        )
                    split_spec = SplitSpec(
                        mode=split_spec["mode"],
                        ports=ports,
                        interleave=interleave,
                    )
                splitter = MultiPortSerializer()
                exposed._split_buffers = splitter.split_tensor(
                    exposed._serialized, split_spec
                )

    # ── Stage 3b: Probe Golden Serialization ──

    def _serialize_probe_golden(self, view: FlattenedKernelView) -> None:
        if not view.probe_points:
            return
        # Probe serialization handled by engine when probe points exist

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

    # ── Stage 5: auto_bind Resolution ──

    def _resolve_auto_binds(self, view: FlattenedKernelView) -> None:
        view._register_bindings = resolve_auto_binds(view)

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

    def _pack_shm(
        self,
        view: FlattenedKernelView,
        commands: list[Command],
        buffer_ids: dict[str, int],
    ) -> bytes:
        shm_alloc = SHMBufferAllocator()

        # Allocate data buffers
        allocated_buffer_ids: set[int] = set()
        for name, exposed in view.exposed_tensors.items():
            bid = buffer_ids[name]
            if bid in allocated_buffer_ids:
                continue
            allocated_buffer_ids.add(bid)
            direction = DIRECTION_ENCODING.get(exposed.direction, 0)
            shm_alloc.allocate(bid, exposed._serialized_size, direction)

        # Probe golden buffers
        next_buffer_id = max(buffer_ids.values(), default=-1) + 1
        for probe in view.probe_points:
            if probe.serialized_golden is not None:
                shm_alloc.allocate(
                    next_buffer_id, len(probe.serialized_golden), 0, flags=0x01
                )
                probe.golden_buffer_id = next_buffer_id
                next_buffer_id += 1

        # Calculate sizes
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

        # Pack control header
        pack_control_header(
            image,
            num_commands=num_commands,
            num_buffers=num_buffers,
            cmd_region_offset=cmd_offset,
            stats_region_offset=stats_offset,
            buf_desc_offset=bufdesc_offset,
            data_region_offset=data_region_offset,
            total_shm_size=total,
        )

        # Pack command slots
        for i, cmd in enumerate(commands):
            pack_command_slot(image, cmd_offset + i * CMD_SLOT_SIZE, cmd)

        # Pack stats entries: LOAD commands pre-marked as COMMITTED
        from vten.spec.models import CommandStatus

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

        # Copy input tensor data
        for name, exposed in view.exposed_tensors.items():
            if exposed._serialized is not None:
                bid = buffer_ids[name]
                try:
                    desc = shm_alloc.get_descriptor(bid)
                except KeyError:
                    continue
                start = data_region_offset + desc.data_offset
                image[start : start + len(exposed._serialized)] = (
                    exposed._serialized
                )

        # Copy probe golden data
        for probe in view.probe_points:
            if probe.serialized_golden is not None and probe.golden_buffer_id is not None:
                desc = shm_alloc.get_descriptor(probe.golden_buffer_id)
                start = data_region_offset + desc.data_offset
                image[start : start + len(probe.serialized_golden)] = (
                    probe.serialized_golden
                )

        return bytes(image)
