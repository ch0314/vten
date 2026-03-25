"""ExecutionContext — User-Facing API.

Spec reference: 02_runtime_engine.md §3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.errors import VerificationError
from vten.spec.models import Direction, OpKind

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

    # ── Shorthands ──

    def send_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.SEND_TENSOR, tensor=tensor, dep=dep)

    def recv_tensor(self, tensor, dep=None) -> OperationHandle:
        return self._record(OpKind.RECV_TENSOR, tensor=tensor, dep=dep)

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
            buffer_id = compiled.buffer_ids[name]
            raw_bytes = backend_result.read_buffer(buffer_id)
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

    # ── Verification internals ──

    def _verify_immediate(self, op_handle, golden) -> None:
        """Eager verification: read HW output from SHM and compare to golden."""
        from vten.runtime.serializer import StreamSerializer

        compiled = self._last_compiled
        backend_result = self._last_backend_result

        tensor_name = op_handle.op.tensor.name
        buffer_id = compiled.buffer_ids[tensor_name]

        raw_bytes = backend_result.read_buffer(buffer_id)
        if not raw_bytes:
            raise VerificationError(
                f"No data returned for tensor '{tensor_name}' "
                f"(buffer_id={buffer_id}). SHM may have been cleaned up.",
                tensor=tensor_name,
            )

        exposed = compiled.flattened_view.exposed_tensors[tensor_name]
        iface = compiled.flattened_view.top_spec.get_interface(
            exposed.top_interface
        )
        packing = iface.packing
        if packing is None:
            raise VerificationError(
                f"No packing scheme for interface '{exposed.top_interface}'",
                tensor=tensor_name,
            )

        serializer = StreamSerializer(packing)
        hw_output = serializer.deserialize(
            raw_bytes,
            exposed.origin_tensor._element_count,
            exposed.origin_tensor._resolved_shape,
            dtype=golden.dtype if golden is not None else None,
        )

        if not self._compare(hw_output, golden):
            raise VerificationError(
                tensor=tensor_name,
                shape=exposed.origin_tensor._resolved_shape,
                max_diff=self._max_diff(hw_output, golden),
            )

    def _run_deferred_verifications(self) -> int:
        """Execute all deferred VerificationTasks after run().

        Returns count of verifications executed.
        """
        count = len(self._verifications)
        for task in self._verifications:
            self._verify_immediate(task.op_handle, task.golden)
        self._verifications.clear()
        return count

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

    def run(self) -> BatchResult:
        """Compile pending ops → submit → wait → return BatchResult."""
        from vten.runtime.engine import RuntimeEngine

        engine = RuntimeEngine(
            kernels=self._kernels,
            ops=self._pending_ops,
            project_params=self._project_params,
            alias_registry=self._alias_registry,
        )
        compiled = engine.compile()
        self._last_compiled = compiled
        self._pending_ops = []

        if self._backend is not None:
            self._backend.submit(
                shm_image=compiled.shm_image,
                bfm_configs=compiled.bfm_configs,
            )
            backend_result = self._backend.wait()
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
            if self._verifications:
                verification_count = self._run_deferred_verifications()

            return BatchResult(
                verification_count=verification_count,
                status=status,
                total_cycles=total_cycles,
                per_command_stats=per_cmd_stats,
                output_tensors=output_tensors,
            )

        return BatchResult(status="DONE")

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
