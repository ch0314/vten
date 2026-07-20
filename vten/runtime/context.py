"""ExecutionContext — User-Facing API.

Spec reference: 02_runtime_engine.md §3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vten.dsl.operations import Operation, OperationHandle
from vten.errors import VerificationError
from vten.spec.models import OpKind

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.kernel.tensor import Tensor
    from vten.runtime.kernel_view import KernelInstance


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
    compiled_result: object = None  # CompiledResult — carries IR command metadata



# ── ExecutionContext ──


class ExecutionContext:
    """User-facing API for recording and executing DSL operations."""

    def __init__(
        self,
        backend: object | None = None,
        project_params: dict | None = None,
        mode: str = "verification",
        project_dir: Path | None = None,
    ) -> None:
        self._pending_ops: list[Operation] = []
        self._kernels: dict[str, KernelInstance] = {}
        self._backend = backend
        self._project_params = project_params or {}
        self._project_dir = project_dir
        self._mode = mode  # "verification" or "inference"
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
        # Declarative probe support
        self._declarative_probes: list[str] = []
        self._internal_probe_requests: list[tuple[str, str]] = []
        # (Session state removed — backend manages its own session lifecycle)

    def instantiate(self, kernel_class: type, spec=None, **params) -> KernelInstance:
        """Create and initialize a kernel instance with eager resolution."""
        from vten.runtime.kernel_view import KernelInstance

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
        instance.initialize(self._project_params, project_dir=self._project_dir)
        self._kernels[instance.name] = instance
        return instance

    # ── Data: Host ↔ DUT ──

    def push_tensor(self, tensor, dep=None, probe=False) -> OperationHandle:
        """Provide tensor data to DUT. Generates LOAD + PUSH IR commands."""
        if self._mode == "inference" and tensor.name in self._bound_bos:
            return self._record(OpKind.PUSH_TENSOR, tensor=tensor, dep=dep,
                                probe=probe, _device_resident=True)
        return self._record(
            OpKind.PUSH_TENSOR, tensor=tensor, dep=dep, probe=probe
        )

    def pull_tensor(
        self, tensor, dep=None, probe=False,
        chunks: int | list[int] | None = None,
    ) -> OperationHandle | list[OperationHandle]:
        """Capture tensor data from DUT. Generates PULL + STORE IR commands.

        Args:
            chunks: Split the receive into multiple PULL command groups.
                int → N equal-sized chunks along the first axis.
                list[int] → explicit per-chunk element counts.
                Returns list[OperationHandle] when chunks is specified.
        """
        if chunks is None:
            return self._record(
                OpKind.PULL_TENSOR, tensor=tensor, dep=dep, probe=probe
            )

        if isinstance(chunks, int):
            chunk_total = chunks
        else:
            chunk_total = len(chunks)

        handles: list[OperationHandle] = []
        for i in range(chunk_total):
            h = self._record(
                OpKind.PULL_TENSOR,
                tensor=tensor,
                dep=dep,
                probe=probe,
                chunk_index=i,
                chunk_total=chunk_total,
                chunks_spec=chunks,
            )
            handles.append(h)
        return handles

    # ── L3: Control ──

    def write_register(self, register, fields: dict, dep=None) -> OperationHandle:
        """Write values to control register fields.

        Args:
            register: RegisterHandle from kernel (e.g. ``self.ctrl``).
            fields: Field-value pairs (e.g. ``{"vsync": 1}``).
            dep: Operation(s) that must complete before this write.
        """
        return self._record(
            OpKind.WRITE_REGISTER,
            register_interface=register.interface_name,
            register_fields=fields,
            dep=dep,
        )

    def read_register(self, register, field_name: str, dep=None) -> OperationHandle:
        """Read a single register field value from DUT.

        Args:
            register: RegisterHandle from kernel.
            field_name: Name of the field to read.
            dep: Operation(s) that must complete before this read.
        """
        return self._record(
            OpKind.READ_REGISTER,
            register_interface=register.interface_name,
            register_field_name=field_name,
            dep=dep,
        )

    def poll_register(
        self, register, field_name: str, *, expected: int | None = None, dep=None,
    ) -> OperationHandle:
        """Poll a register field until it matches expected value.

        Args:
            register: RegisterHandle from kernel.
            field_name: Name of the field to poll.
            expected: Value to wait for (default: 1 for status flags).
            dep: Operation(s) that must complete before polling starts.
        """
        return self._record(
            OpKind.POLL_REGISTER,
            register_interface=register.interface_name,
            register_field_name=field_name,
            poll_expected=expected,
            dep=dep,
        )

    def configure(self, kernel, dep=None) -> OperationHandle:
        """Write all auto-bind register values for a kernel.

        Emits WRITE_REG commands for all registers with auto_bind specs.
        Typically the first operation in a kernel's run() method.
        """
        resolved = getattr(kernel, "_kernel_instance", kernel)
        return self._record(OpKind.CONFIGURE, kernel=resolved, dep=dep)

    def barrier(self) -> OperationHandle:
        """Insert a barrier: all prior operations must complete before any later ones."""
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
                ctx.push_tensor(ki.get_tensor("in"))
                ctx.pull_tensor(ki.get_tensor("out"))
                ctx.config_boundary()
            result = ctx.run()  # single batch, all configs
        """
        self._config_boundaries.append(len(self._pending_ops))
        self._config_kernels.append(dict(self._kernels))
        self._config_params.append(dict(self._project_params))
        self._current_config_group += 1
        # Reset kernels for next group (new instantiate calls go to new group)
        self._kernels = {}

    # ── (send_tensor/recv_tensor/verify removed — use push_tensor/pull_tensor) ──

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
        from vten.runtime.probe_manager import collect_probe_golden_tensors
        return collect_probe_golden_tensors(self._pending_ops)

    # ── Declarative Probes ──

    def _register_declarative_probes(self, probes: list[str]) -> None:
        """Store declarative probe specs for processing at run() time."""
        self._declarative_probes = list(probes)

    def _apply_declarative_probes(self) -> None:
        from vten.runtime.probe_manager import apply_declarative_probes
        if not self._declarative_probes:
            return
        reqs = apply_declarative_probes(self._declarative_probes, self._pending_ops)
        self._internal_probe_requests.extend(reqs)

    def _resolve_internal_probe_golden(self) -> None:
        from vten.runtime.probe_manager import resolve_internal_probe_golden
        resolve_internal_probe_golden(
            self._internal_probe_requests,
            self._kernels,
            self._internal_probe_golden,
        )

    # ── Inference mode: device buffer binding ──

    def bind_device_buffer(self, tensor: Tensor, bo: object) -> None:
        """Bind an existing device BO to a tensor (inference mode).

        When a tensor has a bound BO, send_tensor() skips LOAD+PUSH
        and the BO is injected into CompiledResult.prebound_buffers.
        """
        self._bound_bos[tensor.name] = bo

    # ── Buffer Aliasing ──

    def alias(self, src, dst) -> None:
        """Alias dst tensor to share src's device buffer (zero-copy reuse)."""
        self._alias_registry.register(src, dst)

    # ── Output tensor reading ──

    def _read_output_tensors(
        self, compiled: object, backend_result: object,
    ) -> dict:
        from vten.runtime.output_reader import read_output_tensors
        from vten.backend.base import CompileTarget
        is_hw = (self._backend is not None
                 and self._backend.compile_target == CompileTarget.HW)
        get_bo = (
            self._backend.get_buffer_object
            if is_hw and self._backend is not None else None
        )
        return read_output_tensors(
            compiled, backend_result,
            is_hw=is_hw,
            get_buffer_object=get_bo,
        )

    @staticmethod
    def _make_deserialize_fn(view: object, exposed: object):
        from vten.runtime.output_reader import make_deserialize_fn
        return make_deserialize_fn(view, exposed)

    # ── Auto-verify ──

    def _auto_verify_all(
        self,
        compiled,
        output_tensors,
        *,
        lsb_tolerance: int | dict[str, int] = 0,
    ) -> tuple[int, list]:
        """Auto-verify all DEV_TO_HOST tensors against forward() golden.

        Uses compute_golden_outputs() for logical-format comparison.
        Stores golden on output tensors for inference chain propagation.

        Args:
            lsb_tolerance: Opt-in integer-LSB tolerance — an int applied to
                all outputs, or a dict tensor-name → int (missing names stay
                bit-exact). Default 0 keeps exact comparison. The interface's
                declared QuantSpec (if any) is fetched automatically, but
                purely for report enrichment — it never loosens comparison.

        Returns (count, list[VerificationResult]).
        """
        from vten.runtime.reporting import VerificationResult
        from vten.runtime.golden import compute_golden_outputs
        from vten.verifier import check_match

        view = compiled.flattened_view
        results: list[VerificationResult] = []
        first_error: VerificationError | None = None

        # Collect golden from all registered kernels
        golden_map: dict[str, torch.Tensor] = {}
        for ki in self._kernels.values():
            inst = ki.kernel_class_instance
            if inst is None:
                continue
            try:
                kg = compute_golden_outputs(
                    inst, view, buffer_ids=compiled.buffer_ids,
                )
                golden_map.update(kg)
            except Exception:
                logger.debug("compute_golden_outputs failed for %s", ki, exc_info=True)

        for name, out_tensor in output_tensors.items():
            golden = golden_map.get(name)
            if golden is None:
                continue
            hw = out_tensor.data
            if hw is None:
                continue

            if isinstance(lsb_tolerance, dict):
                tol = int(lsb_tolerance.get(name, 0))
            else:
                tol = int(lsb_tolerance or 0)
            quant = view.quant_for_tensor(name) if view is not None else None

            try:
                max_lsb_err = check_match(
                    name, hw.flatten(), golden.flatten(),
                    shape=tuple(golden.shape),
                    lsb_tol=tol, quant=quant,
                )
                results.append(VerificationResult(
                    tensor_name=name,
                    passed=True,
                    max_lsb_err=max_lsb_err,
                ))
                out_tensor.golden = golden
            except VerificationError as e:
                results.append(VerificationResult(
                    tensor_name=name,
                    passed=False,
                    max_diff=e.max_diff,
                    shape=e.shape,
                    max_lsb_err=e.max_lsb_err,
                ))
                if first_error is None:
                    first_error = e

        count = len(results)

        if first_error is not None:
            first_error.context["verification_results"] = results
            raise first_error

        return count, results

    # ── Execution ──

    def _compile_multi_config(self) -> object:
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
                project_dir=self._project_dir,
            ))

        return RuntimeEngine.compile_multi(engines)

    def run(
        self,
        *,
        verify: bool = False,
        lsb_tolerance: int | dict[str, int] = 0,
    ) -> ExecutionResult:
        """Compile pending ops → submit → wait → return ExecutionResult.

        Args:
            verify: If True, auto-verify all DEV_TO_HOST tensors against
                golden computed from kernel forward().
            lsb_tolerance: Opt-in integer-LSB tolerance for verification —
                an int for all outputs or a dict tensor-name → int.
                Default 0 keeps integer comparison bit-exact.
        """
        from vten.runtime.engine import RuntimeEngine

        # Apply declarative probes (post-hoc annotation)
        self._apply_declarative_probes()
        self._resolve_internal_probe_golden()

        logger.log(5, "ExecutionContext.run(): %d pending ops", len(self._pending_ops))

        if self._config_boundaries:
            compiled = self._compile_multi_config()
        else:
            # CPU backend: skip serialize/deserialize for speed
            from vten.backend.cpu import CpuBackend
            skip_ser = isinstance(self._backend, CpuBackend)

            engine = RuntimeEngine(
                kernels=self._kernels,
                ops=self._pending_ops,
                project_params=self._project_params,
                alias_registry=self._alias_registry,
                quiet=(self._mode == "inference"),
                project_dir=self._project_dir,
                skip_serialize=skip_ser,
            )
            probe_golden_tensors = self._collect_probe_golden_tensors()
            compiled = engine.compile(
                probe_golden_tensors=probe_golden_tensors or None,
                internal_probe_golden=self._internal_probe_golden or None,
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
                logger.debug("_read_output_tensors skipped: %s", e)
                output_tensors = {}

            # Auto-verify all D2H tensors when verify=True
            verification_count = 0
            verification_results: list = []
            if verify:
                verification_count, verification_results = (
                    self._auto_verify_all(
                        compiled, output_tensors,
                        lsb_tolerance=lsb_tolerance,
                    )
                )
                passed = sum(1 for v in verification_results if getattr(v, 'passed', False))
                if passed > 0:
                    names = [v.tensor_name for v in verification_results if getattr(v, 'passed', False)]
                    logger.info("  verify: %d/%d PASS (%s)", passed, verification_count,
                                ", ".join(names))

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
                compiled_result=compiled,
            )

        return ExecutionResult(status="DONE")

    # ── Session lifecycle ──

    def close(self) -> None:
        """No-op. Backend lifecycle is managed by the caller via cleanup()."""
        pass

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
