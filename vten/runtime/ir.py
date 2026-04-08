"""Stage 6: IR Lowering.

Command dataclass and lowering functions.

Spec reference: 00_data_models.md §9, 02_runtime_engine.md §12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.errors import CompilationError, DependencyLimitError
from vten.spec.models import (
    OpCode,
    OpKind,
    Protocol,
    Role,
)
from vten.runtime.binder import parse_bit_range

if TYPE_CHECKING:
    from vten.dsl.operations import Operation, OperationHandle
    from vten.runtime.flattener import FlattenedKernelView


# ── BFMConfig (from 00_data_models.md §10) ──
# Already defined in spec/models.py? No, it's not there. Define it here.


@dataclass
class BFMConfig:
    """BFM instance configuration."""

    interface_name: str
    protocol: Protocol
    data_width: int = 256
    addr_width: int = 64
    role: str = "slave"
    address_ranges: list[tuple[int, int, int]] = field(default_factory=list)
    poll_interval: int = 1
    poll_timeout: int = 100000


# ── Command ──


@dataclass
class Command:
    """Single IR command. Packed to 64-byte SHM slot."""

    op: OpCode
    cmd_id: int
    interface_id: int = 0
    buffer_id: int = 0
    protocol: Protocol = Protocol.AXI4S
    phys_addr: int = 0
    size: int = 0
    role: Role = Role.MASTER
    dep: list[int] = field(default_factory=list)
    commit_dep: list[int] = field(default_factory=list)
    reg_offset: int = 0
    reg_value: int = 0
    reg_mask: int = 0
    reg_expected: int = 0
    probe: bool = False
    golden_buf: int = 0
    sync: bool = False
    port: str = ""


# ── Role determination ──


def _determine_role(protocol: Protocol, opcode: OpCode) -> Role:
    """Determine BFM role from protocol and opcode."""
    if protocol == Protocol.AXI4L:
        return Role.MASTER
    if protocol == Protocol.AXI4S:
        return Role.MASTER if opcode == OpCode.PUSH else Role.SLAVE
    if protocol == Protocol.AXI4:
        return Role.SLAVE
    raise ValueError(f"Unknown protocol: {protocol}")


# ── IR Lowering Engine ──


class IRLowering:
    """Lowers Operations to Commands."""

    def __init__(
        self,
        view: FlattenedKernelView,
        alias_registry: object | None = None,
    ) -> None:
        self._view = view
        self._alias_registry = alias_registry
        self._buffer_ids: dict[str, int] = {}
        # Pre-populate interface ID map using spec order so it matches
        # codegen's iface_to_bfm mapping.
        # Array/split interfaces are expanded to flat element/port names;
        # the logical name maps to the first element's ID for backward compat.
        self._iface_id_map: dict[str, int] = {}
        expanded = view.top_spec.expanded_interface_names()
        for idx, flat_name in enumerate(expanded):
            self._iface_id_map[flat_name] = idx
        # Also map logical array/split names → first element/port ID
        for name, iface in view.top_spec.interfaces.items():
            if iface.array and name not in self._iface_id_map:
                first_flat = iface.array.flat_names(name)[0]
                self._iface_id_map[name] = self._iface_id_map[first_flat]
            elif (iface.split and isinstance(iface.split, dict)
                  and "ports" in iface.split and name not in self._iface_id_map):
                first_port = iface.split["ports"][0]["name"]
                self._iface_id_map[name] = self._iface_id_map[first_port]
        self._next_iface_id = len(expanded)

    def lower(
        self,
        ops: list[Operation],
        *,
        cmd_id_start: int = 0,
        buffer_id_start: int = 0,
    ) -> tuple[list[Command], dict[str, int]]:
        """Lower all operations to commands. Returns (commands, buffer_ids).

        Args:
            ops: Operations to lower.
            cmd_id_start: Starting cmd_id offset (for multi-config batches).
            buffer_id_start: Starting buffer_id offset (for multi-config batches).
        """
        self._buffer_ids = self._allocate_buffer_ids(
            ops, buffer_id_start=buffer_id_start,
        )
        commands: list[Command] = []
        next_cmd_id = cmd_id_start
        op_to_cmd_range: dict[int, tuple[int, int]] = {}

        for op in ops:
            first_cmd_id = next_cmd_id
            new_cmds, next_cmd_id = self._lower_op(
                op, next_cmd_id, op_to_cmd_range
            )

            # Apply commit_dep to last command
            if op.commit_dep and new_cmds:
                commit_dep_ids = self._resolve_deps(
                    op.commit_dep, op_to_cmd_range
                )
                if len(commit_dep_ids) > 4:
                    raise DependencyLimitError(
                        f"commit_dep count {len(commit_dep_ids)} exceeds limit 4"
                    )
                new_cmds[-1].commit_dep = commit_dep_ids

            commands.extend(new_cmds)
            if new_cmds:
                last_cmd_id = next_cmd_id - 1
                op_to_cmd_range[id(op)] = (first_cmd_id, last_cmd_id)
            else:
                # Zero-command op: inherit parent's dep targets so dependents
                # resolve to the same commands this op depended on, avoiding
                # self-referencing phantom cmd_ids.
                parent_ids = self._resolve_deps(op.dep, op_to_cmd_range)
                if parent_ids:
                    op_to_cmd_range[id(op)] = (parent_ids[0], parent_ids[-1])
                # else: no deps and no cmds — op effectively invisible

        return commands, self._buffer_ids

    def _allocate_buffer_ids(
        self,
        ops: list[Operation] | None = None,
        *,
        buffer_id_start: int = 0,
    ) -> dict[str, int]:
        buffer_ids: dict[str, int] = {}
        next_id = buffer_id_start

        # Collect chunk info from ops
        chunk_tensors: dict[str, int] = {}  # tensor_name → chunk_total
        if ops:
            for op in ops:
                if op.chunk_total is not None and op.tensor is not None:
                    chunk_tensors[op.tensor.name] = op.chunk_total

        # Two-pass: allocate non-alias tensors first, then resolve aliases
        alias_targets: list[str] = []
        for name, exposed in self._view.exposed_tensors.items():
            if self._alias_registry and self._alias_registry.is_alias_target(name):
                alias_targets.append(name)
            elif name in chunk_tensors:
                # Chunked tensor: allocate per-chunk (and per-array-element) IDs
                n_chunks = chunk_tensors[name]
                if exposed._port_buffers:
                    port_names = list(exposed._port_buffers.keys())
                    for ci in range(n_chunks):
                        for pname in port_names:
                            buffer_ids[f"{name}:chunk_{ci}:{pname}"] = next_id
                            next_id += 1
                    # Logical name → first chunk's first port
                    buffer_ids[name] = buffer_ids[
                        f"{name}:chunk_0:{port_names[0]}"
                    ]
                else:
                    for ci in range(n_chunks):
                        buffer_ids[f"{name}:chunk_{ci}"] = next_id
                        next_id += 1
                    buffer_ids[name] = buffer_ids[f"{name}:chunk_0"]
            elif exposed._port_buffers:
                # Multi-port tensor: one buffer_id per port
                for port_name in exposed._port_buffers:
                    buffer_ids[f"{name}:{port_name}"] = next_id
                    next_id += 1
                # Logical name → first port's ID (for backward compat)
                first_key = f"{name}:{next(iter(exposed._port_buffers))}"
                buffer_ids[name] = buffer_ids[first_key]
            else:
                buffer_ids[name] = next_id
                next_id += 1
        for name in alias_targets:
            src_name = self._alias_registry.get_source(name)
            buffer_ids[name] = buffer_ids[src_name]
        return buffer_ids

    def _get_iface_id(self, iface_name: str) -> int:
        if iface_name not in self._iface_id_map:
            self._iface_id_map[iface_name] = self._next_iface_id
            self._next_iface_id += 1
        return self._iface_id_map[iface_name]

    def _resolve_deps(
        self,
        op_deps: list[OperationHandle],
        op_to_cmd_range: dict[int, tuple[int, int]],
    ) -> list[int]:
        if not op_deps:
            return []
        cmd_deps: list[int] = []
        for dep_handle in op_deps:
            op_id = id(dep_handle.op)
            if op_id in op_to_cmd_range:
                _, last_cmd_id = op_to_cmd_range[op_id]
                cmd_deps.append(last_cmd_id)
        if len(cmd_deps) > 4:
            raise DependencyLimitError(
                f"issue dep count {len(cmd_deps)} exceeds limit 4"
            )
        return cmd_deps

    def _lower_op(
        self,
        op: Operation,
        next_cmd_id: int,
        op_to_cmd_range: dict[int, tuple[int, int]],
    ) -> tuple[list[Command], int]:
        kind = op.kind
        dep_ids = self._resolve_deps(op.dep, op_to_cmd_range)

        if kind == OpKind.PUSH_TENSOR:
            return self._lower_push(op, dep_ids, next_cmd_id)
        elif kind == OpKind.PULL_TENSOR:
            return self._lower_pull(op, dep_ids, next_cmd_id)
        elif kind == OpKind.WRITE_REGISTER:
            return self._lower_write_reg(op, dep_ids, next_cmd_id)
        elif kind == OpKind.READ_REGISTER:
            return self._lower_read_reg(op, dep_ids, next_cmd_id)
        elif kind == OpKind.POLL_REGISTER:
            return self._lower_poll_reg(op, dep_ids, next_cmd_id)
        elif kind == OpKind.CONFIGURE:
            return self._lower_configure(op, dep_ids, next_cmd_id)
        elif kind == OpKind.BARRIER:
            return self._lower_barrier(op, dep_ids, next_cmd_id)
        else:
            raise CompilationError(f"Unknown OpKind: {kind}")

    # ── Individual lowering methods ──

    def _lower_push(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        """Lower PUSH_TENSOR → LOAD + PUSH (alias/skip-aware)."""
        exposed = self._view.exposed_tensors[op.tensor.name]
        iface = self._view.top_spec.get_interface(exposed.top_interface)
        commands: list[Command] = []

        is_alias_target = (
            self._alias_registry
            and self._alias_registry.is_alias_target(exposed.name)
        )
        skip_load = is_alias_target or getattr(op, "_skip_data", False)

        # Multi-port (array or split): per-port LOAD + PUSH
        if exposed._port_buffers:
            base_addr = exposed.address or 0
            addr_offset = 0
            for i, (port_name, port_data) in enumerate(
                exposed._port_buffers.items()
            ):
                bid = self._buffer_ids[f"{exposed.name}:{port_name}"]
                if not skip_load:
                    load_cmd = Command(
                        op=OpCode.LOAD,
                        cmd_id=next_cmd_id,
                        buffer_id=bid,
                        size=len(port_data),
                        dep=dep_ids if i == 0 else [],
                    )
                    commands.append(load_cmd)
                    load_id = next_cmd_id
                    next_cmd_id += 1
                    push_dep = [load_id]
                else:
                    push_dep = dep_ids if i == 0 else []
                push_cmd = Command(
                    op=OpCode.PUSH,
                    cmd_id=next_cmd_id,
                    interface_id=self._get_iface_id(port_name),
                    buffer_id=bid,
                    protocol=iface.protocol,
                    phys_addr=base_addr + addr_offset,
                    size=len(port_data),
                    role=_determine_role(iface.protocol, OpCode.PUSH),
                    probe=op.probe,
                    dep=push_dep,
                )
                commands.append(push_cmd)
                addr_offset += len(port_data)
                next_cmd_id += 1
            return commands, next_cmd_id

        if not skip_load:
            load_cmd = Command(
                op=OpCode.LOAD,
                cmd_id=next_cmd_id,
                buffer_id=self._buffer_ids[exposed.name],
                size=exposed._serialized_size,
                dep=dep_ids,
            )
            commands.append(load_cmd)
            load_id = next_cmd_id
            next_cmd_id += 1
            push_dep = [load_id]
        else:
            push_dep = list(dep_ids)

        push_cmd = Command(
            op=OpCode.PUSH,
            cmd_id=next_cmd_id,
            interface_id=self._get_iface_id(exposed.top_interface),
            buffer_id=self._buffer_ids[exposed.name],
            protocol=iface.protocol,
            phys_addr=exposed.address or 0,
            size=exposed._serialized_size,
            role=_determine_role(iface.protocol, OpCode.PUSH),
            probe=op.probe,
            dep=push_dep,
        )
        commands.append(push_cmd)
        next_cmd_id += 1
        return commands, next_cmd_id

    def _lower_pull(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        """Lower PULL_TENSOR → PULL + STORE (alias/chunk-aware)."""
        exposed = self._view.exposed_tensors[op.tensor.name]
        iface = self._view.top_spec.get_interface(exposed.top_interface)

        # Chunked: delegate to chunk-aware lowering
        if op.chunk_total is not None:
            return self._lower_pull_chunk(
                op, exposed, iface, dep_ids, next_cmd_id,
            )

        commands: list[Command] = []

        is_alias_source = (
            self._alias_registry
            and self._alias_registry.is_alias_source(exposed.name)
        )

        # Multi-port (array or split): per-port PULL + STORE
        if exposed._port_buffers:
            base_addr = exposed.address or 0
            addr_offset = 0
            for i, (port_name, port_data) in enumerate(
                exposed._port_buffers.items()
            ):
                bid = self._buffer_ids[f"{exposed.name}:{port_name}"]
                pull_cmd = Command(
                    op=OpCode.PULL,
                    cmd_id=next_cmd_id,
                    interface_id=self._get_iface_id(port_name),
                    buffer_id=bid,
                    protocol=iface.protocol,
                    phys_addr=base_addr + addr_offset,
                    size=len(port_data),
                    role=_determine_role(iface.protocol, OpCode.PULL),
                    probe=op.probe,
                    dep=dep_ids if i == 0 else [],
                )
                commands.append(pull_cmd)
                addr_offset += len(port_data)
                pull_id = next_cmd_id
                next_cmd_id += 1

                if self._alias_registry:
                    self._alias_registry.record_write_cmd(exposed.name, pull_id)

                if not is_alias_source and iface.protocol != Protocol.AXI4S:
                    store_cmd = Command(
                        op=OpCode.STORE,
                        cmd_id=next_cmd_id,
                        buffer_id=bid,
                        dep=[pull_id],
                    )
                    commands.append(store_cmd)
                    next_cmd_id += 1
            return commands, next_cmd_id

        pull_cmd = Command(
            op=OpCode.PULL,
            cmd_id=next_cmd_id,
            interface_id=self._get_iface_id(exposed.top_interface),
            buffer_id=self._buffer_ids[exposed.name],
            protocol=iface.protocol,
            phys_addr=exposed.address or 0,
            size=exposed._serialized_size,
            role=_determine_role(iface.protocol, OpCode.PULL),
            probe=op.probe,
            dep=dep_ids,
        )
        commands.append(pull_cmd)
        pull_id = next_cmd_id
        next_cmd_id += 1

        if self._alias_registry:
            self._alias_registry.record_write_cmd(exposed.name, pull_id)

        if not is_alias_source and iface.protocol != Protocol.AXI4S:
            store_cmd = Command(
                op=OpCode.STORE,
                cmd_id=next_cmd_id,
                buffer_id=self._buffer_ids[exposed.name],
                dep=[pull_id],
            )
            commands.append(store_cmd)
            next_cmd_id += 1

        return commands, next_cmd_id

    def _lower_write_reg(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        commands: list[Command] = []
        iface_name = op.register_interface
        iface_id = self._get_iface_id(iface_name)

        for reg_name, value in op.register_fields.items():
            reg_spec, abs_offset = self._resolve_register_by_name(
                iface_name, reg_name
            )
            reg_value = self._encode_register_value(reg_spec, reg_name, value)
            commands.append(
                Command(
                    op=OpCode.WRITE_REG,
                    cmd_id=next_cmd_id,
                    interface_id=iface_id,
                    protocol=Protocol.AXI4L,
                    reg_offset=abs_offset,
                    reg_value=reg_value,
                    dep=dep_ids if not commands else [],
                )
            )
            next_cmd_id += 1
        return commands, next_cmd_id

    def _lower_read_reg(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        iface_name = op.register_interface
        iface_id = self._get_iface_id(iface_name)
        _reg_spec, abs_offset = self._resolve_register_by_field_name(
            iface_name, op.register_field_name
        )
        cmd = Command(
            op=OpCode.READ_REG,
            cmd_id=next_cmd_id,
            interface_id=iface_id,
            protocol=Protocol.AXI4L,
            reg_offset=abs_offset,
            dep=dep_ids,
        )
        return [cmd], next_cmd_id + 1

    def _lower_poll_reg(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        iface_name = op.register_interface
        iface_id = self._get_iface_id(iface_name)
        _reg_spec, abs_offset, mask, expected = self._resolve_poll_params(
            iface_name, op.register_field_name, op.poll_expected,
        )
        cmd = Command(
            op=OpCode.POLL_REG,
            cmd_id=next_cmd_id,
            interface_id=iface_id,
            protocol=Protocol.AXI4L,
            reg_offset=abs_offset,
            reg_mask=mask,
            reg_expected=expected,
            dep=dep_ids,
        )
        return [cmd], next_cmd_id + 1

    def _lower_configure(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        commands: list[Command] = []
        register_bindings = self._view._register_bindings or []

        if op.kernel is not None and hasattr(op.kernel, "name"):
            if op.kernel.name != self._view.name:
                kernel_prefix = f"{self._view.name}.{op.kernel.name}"
                register_bindings = [
                    b
                    for b in register_bindings
                    if b.kernel_path.startswith(kernel_prefix)
                ]

        for i, reg_binding in enumerate(register_bindings):
            iface_id = self._get_iface_id(reg_binding.interface_name)
            commands.append(
                Command(
                    op=OpCode.WRITE_REG,
                    cmd_id=next_cmd_id,
                    interface_id=iface_id,
                    protocol=Protocol.AXI4L,
                    reg_offset=reg_binding.absolute_offset,
                    reg_value=reg_binding.resolved_value,
                    dep=dep_ids if i == 0 else [next_cmd_id - 1],
                )
            )
            next_cmd_id += 1
        return commands, next_cmd_id

    def _lower_barrier(
        self, op: Operation, dep_ids: list[int], next_cmd_id: int
    ) -> tuple[list[Command], int]:
        cmd = Command(
            op=OpCode.BARRIER,
            cmd_id=next_cmd_id,
            dep=dep_ids,
            sync=True,
        )
        return [cmd], next_cmd_id + 1

    def _lower_pull_chunk(
        self,
        op: Operation,
        exposed: object,
        iface: object,
        dep_ids: list[int],
        next_cmd_id: int,
    ) -> tuple[list[Command], int]:
        """Lower a single chunk of a chunked recv_tensor operation."""
        commands: list[Command] = []
        ci = op.chunk_index
        n_chunks = op.chunk_total
        name = exposed.name

        is_alias_source = (
            self._alias_registry
            and self._alias_registry.is_alias_source(name)
        )

        # Compute per-chunk size
        chunks_spec = op.chunks_spec
        if isinstance(chunks_spec, list):
            # Explicit per-chunk element counts — need packing info
            # for now derive byte size from fraction of total
            total_elems = sum(chunks_spec)
            chunk_size = (
                exposed._serialized_size * chunks_spec[ci] // total_elems
            )
        else:
            chunk_size = exposed._serialized_size // n_chunks

        if exposed._port_buffers:
            # Chunked + multi-port: split chunk across ports
            port_names = list(exposed._port_buffers.keys())
            n_elems = len(port_names)
            per_elem_size = chunk_size // n_elems
            base_addr = exposed.address or 0

            for j, fname in enumerate(port_names):
                bid = self._buffer_ids[f"{name}:chunk_{ci}:{fname}"]
                pull_cmd = Command(
                    op=OpCode.PULL,
                    cmd_id=next_cmd_id,
                    interface_id=self._get_iface_id(fname),
                    buffer_id=bid,
                    protocol=iface.protocol,
                    phys_addr=base_addr + j * per_elem_size,
                    size=per_elem_size,
                    role=_determine_role(iface.protocol, OpCode.PULL),
                    dep=dep_ids if j == 0 else [],
                )
                commands.append(pull_cmd)
                pull_id = next_cmd_id
                next_cmd_id += 1

                if self._alias_registry:
                    self._alias_registry.record_write_cmd(name, pull_id)

                if not is_alias_source and iface.protocol != Protocol.AXI4S:
                    store_cmd = Command(
                        op=OpCode.STORE,
                        cmd_id=next_cmd_id,
                        buffer_id=bid,
                        dep=[pull_id],
                    )
                    commands.append(store_cmd)
                    next_cmd_id += 1
        else:
            # Chunked non-array: single PULL per chunk
            bid = self._buffer_ids[f"{name}:chunk_{ci}"]
            pull_cmd = Command(
                op=OpCode.PULL,
                cmd_id=next_cmd_id,
                interface_id=self._get_iface_id(exposed.top_interface),
                buffer_id=bid,
                protocol=iface.protocol,
                phys_addr=exposed.address or 0,
                size=chunk_size,
                role=_determine_role(iface.protocol, OpCode.PULL),
                dep=dep_ids,
            )
            commands.append(pull_cmd)
            pull_id = next_cmd_id
            next_cmd_id += 1

            if self._alias_registry:
                self._alias_registry.record_write_cmd(name, pull_id)

            if not is_alias_source and iface.protocol != Protocol.AXI4S:
                store_cmd = Command(
                    op=OpCode.STORE,
                    cmd_id=next_cmd_id,
                    buffer_id=bid,
                    dep=[pull_id],
                )
                commands.append(store_cmd)
                next_cmd_id += 1

        return commands, next_cmd_id

    # ── Register resolution helpers ──

    def _resolve_register_by_name(
        self, iface_name: str, reg_name: str
    ) -> tuple[object, int]:
        """Resolve register by name (1st: register name, 2nd: field name)."""
        # 1st pass: match register name
        for _sub_name, reg_spec, abs_offset in self._view.registers_for_interface(
            iface_name
        ):
            if reg_spec.name == reg_name:
                return reg_spec, abs_offset

        # 2nd pass: match field name
        for _sub_name, reg_spec, abs_offset in self._view.registers_for_interface(
            iface_name
        ):
            if reg_spec.fields and reg_name in reg_spec.fields:
                return reg_spec, abs_offset

        available = [
            r.name for _, r, _ in self._view.registers_for_interface(iface_name)
        ]
        raise CompilationError(
            f"Register or field '{reg_name}' not found in interface "
            f"'{iface_name}'. Available registers: {available}"
        )

    def _resolve_register_by_field_name(
        self, iface_name: str, field_name: str
    ) -> tuple[object, int]:
        """Resolve register by field name."""
        for _sub_name, reg_spec, abs_offset in self._view.registers_for_interface(
            iface_name
        ):
            if reg_spec.fields and field_name in reg_spec.fields:
                return reg_spec, abs_offset

        available_fields: list[str] = []
        for _, r, _ in self._view.registers_for_interface(iface_name):
            if r.fields:
                available_fields.extend(r.fields.keys())
        raise CompilationError(
            f"Field '{field_name}' not found in any register of interface "
            f"'{iface_name}'. Available fields: {available_fields}"
        )

    def _resolve_poll_params(
        self, iface_name: str, field_name: str,
        poll_expected: int | None = None,
    ) -> tuple[object, int, int, int]:
        """Resolve poll_register mask/expected from field bit range."""
        reg_spec, abs_offset = self._resolve_register_by_field_name(
            iface_name, field_name
        )
        bit_range_str = reg_spec.fields[field_name]
        hi, lo = parse_bit_range(bit_range_str)
        mask = ((1 << (hi - lo + 1)) - 1) << lo
        if poll_expected is not None:
            expected = poll_expected << lo
        else:
            expected = mask  # default: all 1s
        return reg_spec, abs_offset, mask, expected

    @staticmethod
    def _encode_register_value(reg_spec, key_name: str, value: int) -> int:
        """Encode register value (full register or field-shifted)."""
        if key_name == reg_spec.name:
            return int(value)
        if reg_spec.fields and key_name in reg_spec.fields:
            hi, lo = parse_bit_range(reg_spec.fields[key_name])
            mask = ((1 << (hi - lo + 1)) - 1) << lo
            return (int(value) << lo) & mask
        return int(value)
