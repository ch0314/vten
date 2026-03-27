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
    golden: torch.Tensor


# ── Result types ──


@dataclass
class BatchResult:
    """Result of ctx.run(). One Batch execution result."""

    status: str
    total_cycles: int = 0
    per_command_stats: list = field(default_factory=list)
    error: object = None
    output_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    verification_count: int = 0
    verification_results: list = field(default_factory=list)  # list[VerificationResult]


# ── ExecutionContext ──


class ExecutionContext:
    """User-facing API for recording and executing DSL operations."""

    def __init__(
        self,
        backend: object | None = None,
        project_params: dict | None = None,
    ) -> None:
        self._pending_ops: list[Operation] = []
        self._kernels: dict[str, KernelInstance] = {}
        self._backend = backend
        self._project_params = project_params or {}
        self._verifications: list[VerificationTask] = []
        self._alias_registry = AliasRegistry()
        self._last_compiled: object | None = None
        self._last_backend_result: object | None = None
        # Multi-config: config group boundaries
        self._config_boundaries: list[int] = []  # indices into _pending_ops
        self._config_kernels: list[dict[str, KernelInstance]] = []  # per-group
        self._config_params: list[dict] = []  # per-group project_params
        self._current_config_group: int = 0  # tracks current group index
        # Session state (multi-batch)
        self._session_open: bool = False

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

    def poll_register(self, register, field_name: str, dep=None) -> OperationHandle:
        return self._record(
            OpKind.POLL_REGISTER,
            register_interface=register.interface_name,
            register_field_name=field_name,
            dep=dep,
        )

    def configure(self, kernel, dep=None) -> OperationHandle:
        return self._record(OpKind.CONFIGURE, kernel=kernel, dep=dep)

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

    def verify(self, op_handle, golden) -> None:
        """Register or execute verification.

        Before run(): deferred — stored as VerificationTask, executed after run().
        After run(): eager — immediately compares HW output vs golden.
        """
        if self._last_compiled is not None:
            self._verify_immediate(op_handle, golden)
        else:
            self._verifications.append(
                VerificationTask(op_handle=op_handle, golden=golden)
            )

    # ── Buffer Aliasing ──

    def alias(self, src, dst) -> None:
        self._alias_registry.register(src, dst)

    # ── Output tensor reading ──

    def _read_output_tensors(
        self, compiled: object, backend_result: object,
    ) -> dict[str, torch.Tensor]:
        """Deserialize DEV_TO_HOST tensors from backend SHM data."""
        from vten.runtime.serializer import StreamSerializer

        output_tensors: dict[str, torch.Tensor] = {}
        for name, exposed in compiled.flattened_view.exposed_tensors.items():
            if exposed.direction != Direction.DEV_TO_HOST:
                continue
            raw_bytes = self._read_tensor_bytes(
                name, exposed, compiled, backend_result,
            )
            if not raw_bytes:
                continue
            try:
                iface = compiled.flattened_view.top_spec.get_interface(
                    exposed.top_interface
                )
            except KeyError:
                continue
            if iface.packing is None:
                continue
            serializer = StreamSerializer(iface.packing)
            output_tensors[name] = serializer.deserialize(
                raw_bytes,
                exposed.origin_tensor._element_count,
                exposed.origin_tensor._resolved_shape,
                dtype=exposed.origin_tensor.dtype,
            )
        return output_tensors

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

        if not self._compare(hw_output, golden):
            raise VerificationError(
                tensor=tensor_name,
                shape=shape,
                max_diff=self._max_diff(hw_output, golden),
            )

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
                self._verify_immediate(task.op_handle, task.golden)
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
        return (hw_output.float() - golden.float()).abs().max().item()

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
            ))

        return RuntimeEngine.compile_multi(engines, target=target)

    def run(self) -> BatchResult:
        """Compile pending ops → submit → wait → return BatchResult."""
        from vten.runtime.engine import RuntimeEngine

        logger.debug("ExecutionContext.run(): %d pending ops", len(self._pending_ops))

        target = self._backend.compile_target if self._backend else "sim"

        if self._config_boundaries:
            compiled = self._compile_multi_config(target)
        else:
            engine = RuntimeEngine(
                kernels=self._kernels,
                ops=self._pending_ops,
                project_params=self._project_params,
                alias_registry=self._alias_registry,
            )
            compiled = engine.compile(target=target)

        self._last_compiled = compiled
        self._pending_ops = []
        self._config_boundaries = []
        self._config_kernels = []
        self._config_params = []
        self._current_config_group = 0

        if self._backend is not None:
            # Session mode: use open_session/submit_batch/wait_batch
            if (
                hasattr(self._backend, "supports_session")
                and self._backend.supports_session
                and self._session_open
            ):
                logger.debug("submitting batch via session")
                self._backend.submit_batch(compiled)
                backend_result = self._backend.wait_batch()
            elif (
                hasattr(self._backend, "supports_session")
                and self._backend.supports_session
                and not self._session_open
            ):
                logger.debug("opening session + first batch")
                self._backend.open_session(compiled)
                backend_result = self._backend.wait_batch()
                self._session_open = True
            else:
                logger.debug("submitting to backend (one-shot)")
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
            output_tensors = self._read_output_tensors(compiled, backend_result)

            # Run deferred verifications before returning
            verification_count = 0
            verification_results: list = []
            if self._verifications:
                logger.debug("running %d deferred verifications", len(self._verifications))
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

            logger.debug("execution complete: status=%s, cycles=%d, verifications=%d/%d",
                         status, total_cycles,
                         sum(1 for v in verification_results if getattr(v, 'passed', False)),
                         verification_count)

            return BatchResult(
                verification_count=verification_count,
                verification_results=verification_results,
                status=status,
                total_cycles=total_cycles,
                per_command_stats=per_cmd_stats,
                output_tensors=output_tensors,
            )

        return BatchResult(status="DONE")

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
