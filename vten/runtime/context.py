"""ExecutionContext — User-Facing API.

Spec reference: 02_runtime_engine.md §3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vten.dsl.operations import Operation, OperationHandle
from vten.spec.models import OpKind

if TYPE_CHECKING:
    import torch

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
        self._verifications.append(
            VerificationTask(op_handle=op_handle, golden=golden)
        )

    # ── Buffer Aliasing ──

    def alias(self, src, dst) -> None:
        self._alias_registry.register(src, dst)

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

            return BatchResult(
                status=status,
                total_cycles=total_cycles,
                per_command_stats=per_cmd_stats,
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
