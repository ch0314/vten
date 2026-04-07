"""ExecutionContext — User-Facing API.

Spec reference: 02_runtime_engine.md §3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.errors import VerificationError
from vten.spec.models import Direction, OpKind

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.kernel.tensor import Tensor
    from vten.runtime.flattener import KernelInstance


# ── AliasRegistry ──


class AliasRegistry:
    """Tracks buffer aliasing between tensors."""

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}  # dst → src
        self._write_cmds: dict[str, int] = {}  # tensor_name → last write cmd_id

    def register(self, src: Tensor, dst: Tensor) -> None:
        self._aliases[dst.name] = src.name

    def is_alias_target(self, name: str) -> bool:
        return name in self._aliases

    def is_alias_source(self, name: str) -> bool:
        return name in self._aliases.values()

    def get_source(self, name: str) -> str:
        return self._aliases[name]

    def record_write_cmd(self, tensor_name: str, cmd_id: int) -> None:
        self._write_cmds[tensor_name] = cmd_id

    def last_write_cmd_id(self, tensor_name: str) -> int | None:
        return self._write_cmds.get(tensor_name)


# ── VerificationTask ──


@dataclass
class VerificationTask:
    op_handle: OperationHandle
    golden: torch.Tensor | None = None


# ── Result types ──


@dataclass
class ExecutionResult:
    """Result of ctx.run(). User-facing execution result.

    Distinct from ``vten.backend.base.BatchResult`` which is the
    backend-layer result with only (status, total_cycles,
    per_command_stats, error).
    """

    status: str
    total_cycles: int = 0
    per_command_stats: list = field(default_factory=list)
    error: object = None
    output_tensors: dict = field(default_factory=dict)  # dict[str, Tensor]
    verification_count: int = 0
    verification_results: list = field(default_factory=list)  # list[VerificationResult]



# ── ExecutionContext ──


class ExecutionContext:
    """User-facing API for recording and executing DSL operations."""

    def __init__(
        self,
        backend: object | None = None,
        project_params: dict | None = None,
        mode: str = "verification",
    ) -> None:
        self._pending_ops: list[Operation] = []
        self._kernels: dict[str, KernelInstance] = {}
        self._backend = backend
        self._project_params = project_params or {}
        self._mode = mode  # "verification" or "inference"
        self._verifications: list[VerificationTask] = []
        self._bound_bos: dict[str, object] = {}  # tensor_name → xrt.bo
        self._alias_registry = AliasRegistry()
        self._last_compiled: object | None = None
        self._last_backend_result: object | None = None
        # Multi-config: config group boundaries
        self._config_boundaries: list[int] = []  # indices into _pending_ops
        self._config_kernels: list[dict[str, KernelInstance]] = []  # per-group
        self._config_params: list[dict] = []  # per-group project_params
        self._current_config_group: int = 0  # tracks current group index
        # Internal probe golden (composite internal wires)
        self._internal_probe_golden: dict[tuple[str, str], torch.Tensor] = {}
        # Auto-golden cache: id(kernel_instance) → forward() result dict
        self._golden_cache: dict[int, dict[str, torch.Tensor]] = {}
        # Declarative probe support
        self._declarative_probes: list[str] = []
        self._internal_probe_requests: list[tuple[str, str]] = []
        # Session state (multi-batch)
        self._session_open: bool = False
        self._batch_count: int = 0

    def instantiate(self, kernel_class: type, spec=None, **params) -> KernelInstance:
        """Create and initialize a kernel instance with eager resolution."""
        from vten.runtime.flattener import KernelInstance

        if spec is None:
            # Try to load from kernel_class.spec
            spec_path = getattr(kernel_class, "spec", "")
            if spec_path:
                from vten.spec.parser import load_kernel_spec

                spec = load_kernel_spec(spec_path)
            else:
                # Create minimal spec
                from vten.spec.models import KernelSpec

                spec = KernelSpec(
                    kernel_name=kernel_class.__name__,
                    rtl_top=kernel_class.__name__,
                )

        instance = KernelInstance(
            name=kernel_class.__name__,
            spec=spec,
            kernel_class=kernel_class,
            runtime_params=params,
        )
        instance.initialize(self._project_params)
        self._kernels[instance.name] = instance
        return instance

    # ── L1: Host ↔ Memory ──

    def load_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.LOAD_TENSOR, tensor=tensor, dep=dep)

    def store_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.STORE_TENSOR, tensor=tensor, dep=dep)

    # ── L2: Accel ↔ Memory ──

    def push_tensor(self, tensor, dep=None, probe=False) -> OperationHandle:
        return self._record(
            OpKind.PUSH_TENSOR, tensor=tensor, dep=dep, probe=probe
        )

    def pull_tensor(self, tensor, dep=None, probe=False) -> OperationHandle:
        return self._record(
            OpKind.PULL_TENSOR, tensor=tensor, dep=dep, probe=probe
        )

    # ── L3: Control ──

    def write_register(self, register, fields: dict, dep=None) -> OperationHandle:
        return self._record(
            OpKind.WRITE_REGISTER,
            register_interface=register.interface_name,
            register_fields=fields,
            dep=dep,
        )

    def read_register(self, register, field_name: str, dep=None) -> OperationHandle:
        return self._record(
            OpKind.READ_REGISTER,
            register_interface=register.interface_name,
            register_field_name=field_name,
            dep=dep,
        )

    def poll_register(
        self, register, field_name: str, *, expected: int | None = None, dep=None,
    ) -> OperationHandle:
        return self._record(
            OpKind.POLL_REGISTER,
            register_interface=register.interface_name,
            register_field_name=field_name,
            poll_expected=expected,
            dep=dep,
        )

    def configure(self, kernel, dep=None) -> OperationHandle:
        # If passed a Kernel class instance (from kernel.run(ctx)),
        # resolve to the KernelInstance via back-reference
        resolved = getattr(kernel, "_kernel_instance", kernel)
        return self._record(OpKind.CONFIGURE, kernel=resolved, dep=dep)

    def barrier(self) -> OperationHandle:
        return self._record(OpKind.BARRIER)

    # ── Multi-config ──

    def config_boundary(self) -> None:
        """Mark a config group boundary for single-batch multi-config execution.

        All ops recorded before this call belong to the current config group.
        After run(), groups are compiled with separate pipeline passes (Stages 0-6)
        and merged into a single SHM batch with BARRIER commands between them.

        Usage::

            ctx = ExecutionContext(backend=backend)
            for cfg in configs:
                ki = ctx.instantiate(kernel_class, spec=spec, **cfg)
                ctx.send_tensor(ki.get_tensor("in"))
                ctx.recv_tensor(ki.get_tensor("out"))
                ctx.config_boundary()
            result = ctx.run()  # single batch, all configs
        """
        self._config_boundaries.append(len(self._pending_ops))
        self._config_kernels.append(dict(self._kernels))
        self._config_params.append(dict(self._project_params))
        self._current_config_group += 1
        # Reset kernels for next group (new instantiate calls go to new group)
        self._kernels = {}

    # ── Shorthands ──

    def send_tensor(self, tensor, dep=None) -> OperationHandle:
        if self._mode == "inference" and tensor.name in self._bound_bos:
            # Device BO already present — record no-op placeholder
            return self._record(OpKind.SEND_TENSOR, tensor=tensor, dep=dep,
                                _skip_data=True)
        return self._record(OpKind.SEND_TENSOR, tensor=tensor, dep=dep)

    def recv_tensor(
        self, tensor, dep=None, chunks: int | list[int] | None = None,
    ) -> OperationHandle | list[OperationHandle]:
        """Receive tensor from device.

        Args:
            chunks: Split the receive into multiple PULL command groups.
                int → N equal-sized chunks along the first axis.
                list[int] → explicit per-chunk element counts.
                Split is along the serialized byte stream (C-contiguous
                order = axis 0). Each chunk generates separate BFM
                commands, so tready is naturally deasserted/reasserted
                between chunks.
                Returns list[OperationHandle] when chunks is specified.
                TODO: axis= parameter for arbitrary axis split.

        In inference mode: identical to verification mode — PULL+STORE generated.
        """
        if chunks is None:
            return self._record(OpKind.RECV_TENSOR, tensor=tensor, dep=dep)

        if isinstance(chunks, int):
            chunk_total = chunks
        else:
            chunk_total = len(chunks)

        handles: list[OperationHandle] = []
        for i in range(chunk_total):
            h = self._record(
                OpKind.RECV_TENSOR,
                tensor=tensor,
                dep=dep,
                chunk_index=i,
                chunk_total=chunk_total,
                chunks_spec=chunks,
            )
            handles.append(h)
        return handles

    # ── Verification ──

    def verify(self, op_handle, golden=None) -> None:
        """Register or execute verification.

        Before run(): deferred — stored as VerificationTask, executed after run().
        After run(): eager — immediately compares HW output vs golden.

        If golden is None, auto-golden is computed from the kernel's forward().
        In inference mode: no-op (verification is handled by _verify_outputs).
        """
        if self._mode == "inference":
            return
        if self._last_compiled is not None:
            if golden is None:
                golden = self._compute_auto_golden(op_handle)
            self._verify_immediate(op_handle, golden)
        else:
            self._verifications.append(
                VerificationTask(op_handle=op_handle, golden=golden)
            )

    def set_internal_probe_golden(
        self, sub_kernel_name: str, tensor_name: str, golden: torch.Tensor
    ) -> None:
        """Register golden data for a composite internal probe.

        The golden tensor is the expected data on the internal wire
        (source sub-kernel's output). Used by the passive probe BFM
        for beat-by-beat comparison during simulation.

        Args:
            sub_kernel_name: Source sub-kernel attribute name (e.g. "scale").
            tensor_name: Source tensor name (e.g. "data_out").
            golden: Expected golden tensor data.
        """
        self._internal_probe_golden[(sub_kernel_name, tensor_name)] = golden

    def _collect_probe_golden_tensors(self) -> dict[str, torch.Tensor]:
        """Collect golden tensors for probe-enabled PULL operations.

        Handles two patterns:
        1. ctx.verify(h_pull, golden)  — verify directly on probe PULL op
        2. ctx.verify(h_store, golden) — verify on STORE for the same tensor

        In both cases, the golden tensor is matched by tensor name to the
        probe-enabled PULL operation.

        Returns:
            tensor_name → golden torch.Tensor (serialization done by engine).
        """
        # Step 1: find tensor names with probe-enabled operations
        probe_tensor_names: set[str] = set()
        for op in self._pending_ops:
            if op.probe and op.tensor is not None:
                probe_tensor_names.add(op.tensor.name)

        # Step 2: match verifications to probe tensors by name
        probe_golden: dict[str, torch.Tensor] = {}
        for task in self._verifications:
            op = task.op_handle.op
            if op.tensor is None:
                continue
            tensor_name = op.tensor.name
            if tensor_name in probe_tensor_names and tensor_name not in probe_golden:
                golden = task.golden
                if golden is None:
                    # Auto-golden for probe: compute eagerly
                    golden = self._compute_auto_golden(task.op_handle)
                probe_golden[tensor_name] = golden
        return probe_golden

    # ── Declarative Probes ──

    def _register_declarative_probes(self, probes: list[str]) -> None:
        """Store declarative probe specs for processing at run() time."""
        self._declarative_probes = list(probes)

    def _apply_declarative_probes(self) -> None:
        """Post-hoc annotation: mark ops as probes based on declarative specs.

        For output probes (simple name like "data_out"): set probe=True on
        matching PULL/RECV operations.
        For internal probes (dotted name like "scale.data_out"): store in
        _internal_probe_requests for golden extraction.
        """
        if not self._declarative_probes:
            return
        for probe_spec in self._declarative_probes:
            if "." in probe_spec:
                # Internal probe: "sub_kernel.tensor_name"
                sub, tensor = probe_spec.rsplit(".", 1)
                self._internal_probe_requests.append((sub, tensor))
            else:
                # Output probe: find matching PULL/RECV op
                for op in self._pending_ops:
                    if (
                        op.kind in (OpKind.PULL_TENSOR, OpKind.RECV_TENSOR)
                        and op.tensor is not None
                        and op.tensor.name == probe_spec
                    ):
                        op.probe = True

    def _resolve_internal_probe_golden(self) -> None:
        """Auto-extract internal probe golden from CompositeKernel forward results.

        v2: Uses _sub_kernel_refs and forward() chain instead of golden_provides/pool.
        After forward() has run during ki.run(ctx), we can extract intermediate
        values from the forward chain pool stored on the composite instance.
        """
        if not self._internal_probe_requests:
            return
        for ki in self._kernels.values():
            inst = ki.kernel_class_instance
            pool = getattr(inst, "_golden_pool", None)
            if pool is None:
                continue
            for sub_name, tensor_name in self._internal_probe_requests:
                if (sub_name, tensor_name) in self._internal_probe_golden:
                    continue
                # v2: pool keys are (sub_name, tensor_name) tuples
                key = (sub_name, tensor_name)
                if key in pool:
                    self._internal_probe_golden[key] = pool[key]

    def _compute_shm_flags(self) -> int:
        """Compute SHM control header flags from backend config."""
        from vten.runtime.shm import (
            FLAG_PAUSE_ON_MISMATCH,
            FLAG_WAVEFORM_DUMP,
        )
        flags = 0  # FLAG_STATS_ENABLED is always added by engine
        if self._backend and hasattr(self._backend, "_config"):
            cfg = self._backend._config
            if cfg.get("_waveform"):
                flags |= FLAG_WAVEFORM_DUMP
            if cfg.get("_gui"):
                flags |= FLAG_PAUSE_ON_MISMATCH
        return flags

    # ── Inference mode: device buffer binding ──

    def bind_device_buffer(self, tensor: Tensor, bo: object) -> None:
        """Bind an existing device BO to a tensor (inference mode).

        When a tensor has a bound BO, send_tensor() skips LOAD+PUSH
        and the BO is injected into CompiledResult.prebound_buffers.
        """
        self._bound_bos[tensor.name] = bo

    # ── Buffer Aliasing ──

    def alias(self, src, dst) -> None:
        self._alias_registry.register(src, dst)

    # ── Output tensor reading ──

    def _read_output_tensors(
        self, compiled: object, backend_result: object,
    ) -> dict[str, Tensor]:
        """Deserialize DEV_TO_HOST tensors from backend result.

        Returns Tensor objects with:
          - .data = deserialized + unlayouted torch.Tensor
          - BO binding for HW backends (supports .cpu() from device)
          - Metadata (shape, dtype, etc.) from origin tensor
        """
        from vten.kernel.tensor import Tensor as TensorCls
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.serializer import StreamSerializer

        view = compiled.flattened_view
        is_hw = (self._backend is not None
                 and self._backend.compile_target == "hw")

        output_tensors: dict[str, TensorCls] = {}
        for name, exposed in view.exposed_tensors.items():
            if exposed.direction != Direction.DEV_TO_HOST:
                continue

            origin = exposed.origin_tensor
            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
            except KeyError:
                continue
            if iface.packing is None:
                continue

            # Create Tensor wrapper
            t = TensorCls(
                shape=origin._resolved_shape or origin.shape,
                dtype=origin.dtype,
                interface=origin.interface,
                direction=Direction.DEV_TO_HOST,
            )
            t.name = name
            t._resolved_shape = origin._resolved_shape
            t._element_count = origin._element_count

            # Deserialize from backend result
            raw_bytes = self._read_tensor_bytes(
                name, exposed, compiled, backend_result,
            )
            if raw_bytes:
                serializer = StreamSerializer(iface.packing)
                hw_tensor = serializer.deserialize(
                    raw_bytes,
                    origin._element_count,
                    origin._resolved_shape,
                    dtype=origin.dtype,
                )
                t.data = RuntimeEngine._apply_unlayout(view, exposed, hw_tensor)

            # HW backend: bind BO for device-resident access
            if is_hw:
                buffer_id = compiled.buffer_ids.get(name)
                if buffer_id is not None:
                    bo = self._backend.get_buffer_object(buffer_id)
                    if bo is not None:
                        deserialize_fn = self._make_deserialize_fn(view, exposed)
                        bo_size = (bo.size() if hasattr(bo, "size")
                                   else getattr(exposed, "_serialized_size", 0))
                        t._bind_bo(bo, bo_size, deserialize_fn)

            output_tensors[name] = t
        return output_tensors

    @staticmethod
    def _make_deserialize_fn(view: object, exposed: object):
        """Create a bytes → torch.Tensor deserialize+unlayout closure.

        Used by Tensor._bind_bo() for .cpu() deserialization.
        """
        from vten.runtime.engine import RuntimeEngine
        from vten.runtime.serializer import StreamSerializer

        try:
            iface = view.top_spec.get_interface(exposed.top_interface)
        except (KeyError, AttributeError):
            return None
        if iface.packing is None:
            return None

        packing = iface.packing
        origin = exposed.origin_tensor
        element_count = origin._element_count
        shape = origin._resolved_shape
        dtype = origin.dtype

        def _deserialize(raw: bytes) -> torch.Tensor:
            serializer = StreamSerializer(packing)
            hw_tensor = serializer.deserialize(
                raw, element_count, shape, dtype=dtype,
            )
            return RuntimeEngine._apply_unlayout(view, exposed, hw_tensor)

        return _deserialize

    @staticmethod
    def _read_tensor_bytes(
        name: str, exposed, compiled, backend_result,
        buffer_prefix: str = "",
    ) -> bytes:
        """Read raw bytes for a tensor, reassembling array/chunk buffers.

        Args:
            buffer_prefix: Key prefix for multi-config (e.g. "cfg1:").
        """
        prefixed = f"{buffer_prefix}{name}"
        # Detect chunked buffers: look for chunk_0 key pattern
        chunk_0_key = f"{prefixed}:chunk_0"
        is_chunked = any(
            k.startswith(chunk_0_key) for k in compiled.buffer_ids
        )

        if is_chunked:
            return ExecutionContext._read_all_chunk_bytes(
                name, exposed, compiled, backend_result,
                buffer_prefix=buffer_prefix,
            )

        if exposed._port_buffers:
            parts = {}
            for port_name in exposed._port_buffers:
                key = f"{prefixed}:{port_name}"
                bid = compiled.buffer_ids.get(key)
                if bid is None:
                    continue
                data = backend_result.read_buffer(bid)
                if data:
                    parts[port_name] = data
            if exposed._port_mode == "channel_interleave" and parts:
                from vten.runtime.serializer import MultiPortSerializer
                return MultiPortSerializer.reassemble(
                    parts, exposed._interleave_unit
                )
            return b"".join(parts.values())
        buffer_id = compiled.buffer_ids[prefixed]
        return backend_result.read_buffer(buffer_id)

    @staticmethod
    def _read_all_chunk_bytes(
        name: str, exposed, compiled, backend_result,
        buffer_prefix: str = "",
    ) -> bytes:
        """Read and concatenate all chunk buffers for a chunked tensor."""
        prefixed = f"{buffer_prefix}{name}"
        parts: list[bytes] = []
        ci = 0
        while True:
            if exposed._port_buffers:
                # Chunked + array: read per-chunk-per-element
                chunk_parts: list[bytes] = []
                for fname in exposed._port_buffers:
                    key = f"{prefixed}:chunk_{ci}:{fname}"
                    bid = compiled.buffer_ids.get(key)
                    if bid is None:
                        return b"".join(parts)
                    data = backend_result.read_buffer(bid)
                    if data:
                        chunk_parts.append(data)
                parts.extend(chunk_parts)
            else:
                key = f"{prefixed}:chunk_{ci}"
                bid = compiled.buffer_ids.get(key)
                if bid is None:
                    break
                data = backend_result.read_buffer(bid)
                if data:
                    parts.append(data)
            ci += 1
        return b"".join(parts)

    @staticmethod
    def _read_chunk_bytes(
        name: str, exposed, compiled, backend_result,
        chunk_index: int, buffer_prefix: str = "",
    ) -> bytes:
        """Read raw bytes for a single chunk of a chunked tensor."""
        prefixed = f"{buffer_prefix}{name}"
        if exposed._port_buffers:
            parts: list[bytes] = []
            for fname in exposed._port_buffers:
                key = f"{prefixed}:chunk_{chunk_index}:{fname}"
                bid = compiled.buffer_ids.get(key)
                if bid is None:
                    continue
                data = backend_result.read_buffer(bid)
                if data:
                    parts.append(data)
            return b"".join(parts)
        key = f"{prefixed}:chunk_{chunk_index}"
        bid = compiled.buffer_ids[key]
        return backend_result.read_buffer(bid)

    # ── Auto-golden ──

    def _find_kernel_for_tensor(self, tensor_name: str) -> object | None:
        """Find the kernel class instance that owns the given tensor name."""
        for ki in self._kernels.values():
            inst = ki.kernel_class_instance
            if inst is None:
                continue
            try:
                inst.get_tensor(tensor_name)
                return inst
            except AttributeError:
                pass
            # Check auto-exposed tensors (CompositeKernel)
            auto_exposed = getattr(type(inst), "_auto_exposed", {})
            for (_sub, _tname), exposed_name in auto_exposed.items():
                if exposed_name == tensor_name:
                    return inst
        return None

    def _run_forward(self, kernel_inst: object) -> dict[str, torch.Tensor]:
        """Run forward() on a kernel instance, handling Composite vs Simple.

        CompositeKernel: forward() with no args (auto-chain with layout).
        Simple Kernel: collect H2D tensor data, apply layout, forward(**inputs).
        """
        from vten.runtime.golden import run_forward

        return run_forward(kernel_inst)

    def _compute_auto_golden(self, op_handle) -> torch.Tensor:
        """Compute golden tensor automatically from kernel's forward().

        1. Find kernel that owns the tensor
        2. Run forward() (cached per kernel instance)
        3. Extract the relevant output
        4. Handle format conversion (e.g., packed uint8 → int32)
        5. Handle chunk slicing
        """
        from vten.runtime.serializer import StreamSerializer

        op = op_handle.op
        tensor_name = op.tensor.name

        # Find owning kernel
        kernel_inst = self._find_kernel_for_tensor(tensor_name)
        if kernel_inst is None:
            raise VerificationError(
                f"Auto-golden: cannot find kernel for tensor '{tensor_name}'",
                tensor=tensor_name,
            )

        # Run forward() with caching
        cache_key = id(kernel_inst)
        if cache_key not in self._golden_cache:
            self._golden_cache[cache_key] = self._run_forward(kernel_inst)
        fwd_result = self._golden_cache[cache_key]

        if tensor_name not in fwd_result:
            raise VerificationError(
                f"Auto-golden: forward() did not produce '{tensor_name}'. "
                f"Available: {list(fwd_result.keys())}",
                tensor=tensor_name,
            )

        golden = fwd_result[tensor_name]

        # Format conversion: if forward() dtype differs from tensor dtype
        origin_tensor = op.tensor
        target_dtype = origin_tensor.dtype
        if golden.dtype != target_dtype:
            # Check if this is a packed → unpacked case (e.g., uint8 → int32)
            # Try deserializing via the interface's packing scheme
            if self._last_compiled is not None:
                compiled = self._last_compiled
                config_group = getattr(op, "config_group", 0)
                if config_group > 0 and compiled.views and config_group < len(compiled.views):
                    view = compiled.views[config_group]
                else:
                    view = compiled.flattened_view

                exposed = view.exposed_tensors.get(tensor_name)
                if exposed is not None:
                    try:
                        iface = view.top_spec.get_interface(exposed.top_interface)
                        if iface.packing is not None:
                            serializer = StreamSerializer(iface.packing)
                            raw_bytes = serializer.serialize(golden)
                            element_count = origin_tensor._element_count
                            shape = origin_tensor._resolved_shape
                            golden = serializer.deserialize(
                                raw_bytes, element_count, shape,
                                dtype=target_dtype,
                            )
                    except (KeyError, AttributeError):
                        pass

            # Fallback: simple dtype cast
            if golden.dtype != target_dtype:
                golden = golden.to(target_dtype)

        # Flatten for comparison
        golden = golden.flatten()

        # Chunk slicing
        if op.chunk_index is not None:
            total_elems = origin_tensor._element_count
            if isinstance(op.chunks_spec, list):
                chunk_elems = op.chunks_spec[op.chunk_index]
                start = sum(op.chunks_spec[:op.chunk_index])
            else:
                chunk_elems = total_elems // op.chunk_total
                start = op.chunk_index * chunk_elems
            golden = golden[start:start + chunk_elems]

        return golden

    # ── Verification internals ──

    def _verify_immediate(self, op_handle, golden) -> None:
        """Eager verification: read HW output from SHM and compare to golden."""
        from vten.runtime.serializer import StreamSerializer

        compiled = self._last_compiled
        backend_result = self._last_backend_result

        tensor_name = op_handle.op.tensor.name
        op = op_handle.op

        # Multi-config: use correct view and buffer prefix for config group
        config_group = getattr(op, "config_group", 0)
        if config_group > 0 and compiled.views and config_group < len(compiled.views):
            view = compiled.views[config_group]
            buffer_prefix = f"cfg{config_group}:"
        else:
            view = compiled.flattened_view
            buffer_prefix = ""

        exposed = view.exposed_tensors[tensor_name]

        # Chunked: read only this chunk's buffer
        if op.chunk_index is not None:
            raw_bytes = self._read_chunk_bytes(
                tensor_name, exposed, compiled, backend_result,
                op.chunk_index, buffer_prefix=buffer_prefix,
            )
        else:
            raw_bytes = self._read_tensor_bytes(
                tensor_name, exposed, compiled, backend_result,
                buffer_prefix=buffer_prefix,
            )

        if not raw_bytes:
            chunk_label = (
                f" chunk {op.chunk_index}" if op.chunk_index is not None else ""
            )
            raise VerificationError(
                f"No data returned for tensor '{tensor_name}'{chunk_label}. "
                f"SHM may have been cleaned up.",
                tensor=tensor_name,
            )

        iface = view.top_spec.get_interface(
            exposed.top_interface
        )
        packing = iface.packing
        if packing is None:
            raise VerificationError(
                f"No packing scheme for interface '{exposed.top_interface}'",
                tensor=tensor_name,
            )

        serializer = StreamSerializer(packing)

        # For chunked ops, compute per-chunk element count and shape
        if op.chunk_index is not None:
            element_count, shape = self._chunk_element_info(
                exposed, op.chunk_index, op.chunk_total, op.chunks_spec,
            )
        else:
            element_count = exposed.origin_tensor._element_count
            shape = exposed.origin_tensor._resolved_shape

        hw_output = serializer.deserialize(
            raw_bytes,
            element_count,
            shape,
            dtype=golden.dtype if golden is not None else None,
        )

        # Flatten to match golden (which is flattened in _compute_auto_golden)
        hw_output = hw_output.flatten()

        self._check_match(tensor_name, hw_output, golden, shape=shape)

    @staticmethod
    def _chunk_element_info(
        exposed, chunk_index: int, chunk_total: int,
        chunks_spec: int | list[int] | None,
    ) -> tuple[int, tuple[int, ...]]:
        """Compute element count and shape for a single chunk."""
        total_elems = exposed.origin_tensor._element_count
        if isinstance(chunks_spec, list):
            chunk_elems = chunks_spec[chunk_index]
        else:
            chunk_elems = total_elems // chunk_total
        return chunk_elems, (chunk_elems,)

    def _run_deferred_verifications(self) -> tuple[int, list]:
        """Execute all deferred VerificationTasks after run().

        Collects all results before raising on failure.
        Returns (count, list[VerificationResult]).
        """
        from vten.reporting import VerificationResult

        count = len(self._verifications)
        results: list[VerificationResult] = []
        first_error: VerificationError | None = None

        for task in self._verifications:
            tensor_name = task.op_handle.op.tensor.name
            try:
                golden = task.golden
                if golden is None:
                    golden = self._compute_auto_golden(task.op_handle)
                self._verify_immediate(task.op_handle, golden)
                results.append(VerificationResult(
                    tensor_name=tensor_name,
                    passed=True,
                ))
            except VerificationError as e:
                results.append(VerificationResult(
                    tensor_name=tensor_name,
                    passed=False,
                    max_diff=e.max_diff,
                    shape=e.shape,
                ))
                if first_error is None:
                    first_error = e

        self._verifications.clear()

        if first_error is not None:
            # Attach all results to the error for reporting
            first_error.context["verification_results"] = results
            raise first_error

        return count, results

    @staticmethod
    def _compare(hw_output: torch.Tensor, golden: torch.Tensor) -> bool:
        """Element-wise comparison with tolerance."""
        if hw_output.shape != golden.shape:
            return False
        if golden.is_floating_point():
            return torch.allclose(hw_output.float(), golden.float(), atol=1e-6, rtol=1e-5)
        return torch.equal(hw_output, golden)

    @staticmethod
    def _max_diff(hw_output: torch.Tensor, golden: torch.Tensor) -> float:
        """Maximum element-wise absolute difference."""
        a, b = hw_output.flatten().float(), golden.flatten().float()
        n = min(a.numel(), b.numel())
        return (a[:n] - b[:n]).abs().max().item()

    @staticmethod
    def _check_match(
        tensor_name: str,
        hw_output: torch.Tensor,
        golden: torch.Tensor,
        *,
        shape: tuple | None = None,
    ) -> None:
        """Compare HW output against golden, raise VerificationError on mismatch.

        Shared by both TestScenario (deferred verification) and InferenceSession
        (golden verification) paths.
        """
        if ExecutionContext._compare(hw_output, golden):
            return

        max_diff = ExecutionContext._max_diff(hw_output, golden)
        dtype_str = str(golden.dtype).replace("torch.", "")
        diff_mask = hw_output != golden
        diff_indices = diff_mask.nonzero(as_tuple=False)
        n_diff = diff_indices.shape[0]

        # Show first few differing elements
        detail_parts: list[str] = []
        _show = min(n_diff, 4)
        for i in range(_show):
            idx = tuple(diff_indices[i].tolist())
            idx_str = f"[{','.join(str(x) for x in idx)}]"
            detail_parts.append(
                f"  {idx_str}: expected={golden[idx].item()}, "
                f"actual={hw_output[idx].item()}"
            )
        if n_diff > _show:
            detail_parts.append(f"  ... and {n_diff - _show} more elements differ")

        effective_shape = shape or tuple(hw_output.shape)
        msg = (
            f"Verification failed for tensor '{tensor_name}': "
            f"shape={effective_shape}, dtype={dtype_str}, max_diff={max_diff}, "
            f"{n_diff}/{hw_output.numel()} elements differ"
        )
        detail = "\n".join(detail_parts)
        if detail:
            msg += f"\n{detail}"

        raise VerificationError(
            msg,
            tensor=tensor_name,
            shape=effective_shape,
            max_diff=max_diff,
        )

    # ── Execution ──

    def _compile_multi_config(self, target: str) -> object:
        """Split pending ops by config boundaries and compile as multi-config batch."""
        from vten.runtime.engine import RuntimeEngine

        # Build op groups: split _pending_ops at each boundary
        boundaries = self._config_boundaries
        op_groups: list[list] = []
        start = 0
        for b in boundaries:
            op_groups.append(self._pending_ops[start:b])
            start = b
        # Remaining ops after last boundary (if any)
        if start < len(self._pending_ops):
            op_groups.append(self._pending_ops[start:])
            # This group uses current self._kernels
            self._config_kernels.append(dict(self._kernels))
            self._config_params.append(dict(self._project_params))

        # Build one RuntimeEngine per group
        engines = []
        for i, ops in enumerate(op_groups):
            kernels = self._config_kernels[i] if i < len(self._config_kernels) else self._kernels
            params = self._config_params[i] if i < len(self._config_params) else self._project_params
            engines.append(RuntimeEngine(
                kernels=kernels,
                ops=ops,
                project_params=params,
                alias_registry=self._alias_registry,
                quiet=(self._mode == "inference"),
            ))

        return RuntimeEngine.compile_multi(engines, target=target)

    def run(self) -> ExecutionResult:
        """Compile pending ops → submit → wait → return ExecutionResult."""
        from vten.runtime.engine import RuntimeEngine

        # Apply declarative probes (post-hoc annotation)
        self._apply_declarative_probes()
        self._resolve_internal_probe_golden()

        logger.log(5, "ExecutionContext.run(): %d pending ops", len(self._pending_ops))

        target = self._backend.compile_target if self._backend else "sim"

        if self._config_boundaries:
            compiled = self._compile_multi_config(target)
        else:
            engine = RuntimeEngine(
                kernels=self._kernels,
                ops=self._pending_ops,
                project_params=self._project_params,
                alias_registry=self._alias_registry,
                quiet=(self._mode == "inference"),
            )
            probe_golden_tensors = self._collect_probe_golden_tensors()
            shm_flags = self._compute_shm_flags()
            compiled = engine.compile(
                target=target,
                probe_golden_tensors=probe_golden_tensors or None,
                internal_probe_golden=self._internal_probe_golden or None,
                flags=shm_flags,
            )

        self._last_compiled = compiled
        self._pending_ops = []
        self._config_boundaries = []
        self._config_kernels = []
        self._config_params = []
        self._current_config_group = 0

        # Inject prebound device buffers (inference mode)
        if self._bound_bos:
            for tensor_name, bo in self._bound_bos.items():
                bid = compiled.buffer_ids.get(tensor_name)
                if bid is not None:
                    compiled.prebound_buffers[bid] = bo
                    logger.debug("prebound %s → bid=%d", tensor_name, bid)

        if self._backend is not None:
            # Session mode: use open_session/submit_batch/wait_batch
            if (
                hasattr(self._backend, "supports_session")
                and self._backend.supports_session
                and self._session_open
            ):
                self._batch_count += 1
                logger.debug("──── batch #%d submit (session reuse) ────",
                             self._batch_count)
                self._backend.submit_batch(compiled)
                backend_result = self._backend.wait_batch()
            elif (
                hasattr(self._backend, "supports_session")
                and self._backend.supports_session
                and not self._session_open
            ):
                self._batch_count += 1
                logger.debug("──── batch #%d submit (new session) ────",
                             self._batch_count)
                self._backend.open_session(compiled)
                backend_result = self._backend.wait_batch()
                self._session_open = True
            else:
                self._batch_count += 1
                logger.debug("──── batch #%d submit (one-shot) ────",
                             self._batch_count)
                backend_result = self._backend.execute(compiled)
            self._last_backend_result = backend_result

            # Build BatchResult from backend stats
            total_cycles = 0
            per_cmd_stats = []
            if hasattr(backend_result, "stats") and backend_result.stats:
                per_cmd_stats = list(backend_result.stats)
                max_cycle = max(
                    (s.commit_cycle for s in backend_result.stats if s.commit_cycle),
                    default=0,
                )
                total_cycles = max_cycle

            status = "DONE"
            if hasattr(backend_result, "error_code") and backend_result.error_code:
                status = "ERROR"

            # Read output tensors from backend
            try:
                output_tensors = self._read_output_tensors(compiled, backend_result)
            except (RuntimeError, ValueError) as e:
                # Inference HW mode may have size mismatches when tensor
                # _element_count doesn't match STORE data size. Fall back to
                # empty dict — inference.py's _wrap_outputs handles HW outputs
                # via get_buffer_object() or output_buffers directly.
                logger.debug("_read_output_tensors skipped: %s", e)
                output_tensors = {}

            # Run deferred verifications before returning
            verification_count = 0
            verification_results: list = []
            if self._verifications:
                logger.log(5, "running %d deferred verifications", len(self._verifications))
                try:
                    verification_count, verification_results = (
                        self._run_deferred_verifications()
                    )
                except VerificationError as e:
                    verification_count = len(
                        e.context.get("verification_results", [])
                    )
                    verification_results = e.context.get(
                        "verification_results", []
                    )
                    raise

            self._golden_cache.clear()

            logger.debug("execution complete: status=%s, cycles=%d, verifications=%d/%d",
                         status, total_cycles,
                         sum(1 for v in verification_results if getattr(v, 'passed', False)),
                         verification_count)

            return ExecutionResult(
                verification_count=verification_count,
                verification_results=verification_results,
                status=status,
                total_cycles=total_cycles,
                per_command_stats=per_cmd_stats,
                output_tensors=output_tensors,
            )

        return ExecutionResult(status="DONE")

    # ── Session lifecycle ──

    def close(self) -> None:
        """Close the backend session if one is open. Idempotent."""
        if self._session_open and self._backend is not None:
            if hasattr(self._backend, "close_session"):
                self._backend.close_session()
            self._session_open = False

    def __enter__(self) -> ExecutionContext:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Internal ──

    def _record(self, kind: OpKind, **kwargs) -> OperationHandle:
        dep = kwargs.pop("dep", None)
        op = Operation(
            kind=kind,
            dep=self._normalize_deps(dep),
            commit_dep=[],
            probe=kwargs.pop("probe", False),
            sync=kwargs.pop("sync", False),
            golden=None,
            verify=False,
            config_group=self._current_config_group,
            **kwargs,
        )
        self._pending_ops.append(op)
        return OperationHandle(op=op)

    @staticmethod
    def _normalize_deps(dep) -> list[OperationHandle]:
        if dep is None:
            return []
        if isinstance(dep, OperationHandle):
            return [dep]
        return list(dep)
