"""Inference API — Kernel-Granular Eager Execution.

Provides InferenceSession (eager executor) and InferenceModule (nn.Module wrapper)
for running verified kernels on real FPGA hardware.

Spec reference: 11_inference_api.md
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any

import torch

from vten.log import format_elapsed, format_size

from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.backend.base import Backend
    from vten.kernel.base import Kernel


class InferenceSession:
    """Kernel-granular eager executor.

    Each run() executes a single kernel and returns Tensor(on_device=True).
    Python controls the data flow between kernels (like PyTorch eager mode).

    Usage::

        # Auto-discover xclbin from project build structure
        session = InferenceSession(
            kernel="npu_pipeline", backend="xrt", target="hw_emu",
        )

        # Or from explicit xclbin path
        session = InferenceSession.from_xclbin("path/to/design.xclbin")

        r1 = session.run(NpuKernel, inputs={"ifm": x}, **L1)
        r2 = session.run(NpuKernel, inputs={"ifm": r1["ofm"]}, **L2)
        y = r2["ofm"].cpu()

        # With per-layer verification:
        r1 = session.run(NpuKernel, inputs={"ifm": x}, verify=True, **L1)
        r2 = session.run(NpuKernel, inputs={"ifm": r1["ofm"]}, verify=True, **L2)
    """

    def __init__(
        self,
        backend: Backend | str = "xrt",
        base_params: dict | None = None,
        *,
        kernel: str | None = None,
        target: str = "hw",
        project_dir: str = ".",
        log_level: str | None = None,
    ) -> None:
        """Create an inference session.

        Args:
            backend: Backend instance, or backend name ("xrt") for auto-setup.
            base_params: Default parameters for all run() calls.
            kernel: Kernel name for xclbin auto-discovery (e.g. "npu_pipeline").
            target: "hw" or "hw_emu" (used with string backend).
            project_dir: Project root containing vten.toml (default: ".").
            log_level: Log level (e.g. "DEBUG", "INFO"). If None, auto-configures
                to INFO when no vten handlers exist.
        """
        if isinstance(backend, str):
            backend, project_config = self._create_backend(
                backend, kernel=kernel, target=target, project_dir=project_dir,
            )
            # Auto-inject build_params from vten.toml so kernels see Ti, To, etc.
            build_params = project_config.get("build_params")
            if build_params:
                merged = dict(base_params or {})
                merged.setdefault("build_params", {}).update(build_params)
                base_params = merged
        self._backend = backend
        self._base_params = base_params or {}
        self._run_count = 0  # tracks run() calls for logging
        # Auto-configure vten logging if no handlers set (library user mode)
        vten_root = logging.getLogger("vten")
        if log_level or not vten_root.handlers:
            from vten.log import setup_logging
            setup_logging(level=log_level or "INFO")
        # Enable persistent mode for BO pool reuse
        if hasattr(backend, "_persistent"):
            backend._persistent = True

    @staticmethod
    def _create_backend(
        backend_name: str,
        *,
        kernel: str | None = None,
        target: str = "hw",
        project_dir: str = ".",
    ) -> tuple[Backend, dict]:
        """Create backend from name with auto-discovery.

        Returns (backend, project_config) so caller can extract build_params.
        """
        from pathlib import Path

        if backend_name != "xrt":
            raise ValueError(f"unsupported inference backend: {backend_name!r}")

        from vten.backend.xrt import XrtBackend

        # Load vten.toml if it exists
        project = Path(project_dir).resolve()
        project_config: dict[str, Any] = {}
        toml_path = project / "vten.toml"
        if toml_path.exists():
            from vten.cli.config import load_project_config
            project_config = load_project_config(project)

        # Inject backend target
        project_config.setdefault("backend", {}).setdefault("xrt", {})["target"] = target

        # Auto-discover xclbin from project build structure
        if kernel is not None:
            kernel_dir = project / "kernels" / kernel
            project_config["_kernel_build_dir"] = str(kernel_dir / "build")

        return XrtBackend(project_config, persistent=True), project_config

    @classmethod
    def from_xclbin(
        cls,
        xclbin_path: str,
        *,
        target: str = "hw",
        base_params: dict | None = None,
    ) -> InferenceSession:
        """Create session from explicit xclbin path.

        Args:
            xclbin_path: Path to .xclbin file.
            target: "hw" for real FPGA, "hw_emu" for hardware emulation.
            base_params: Default parameters for all run() calls.
        """
        from vten.backend.xrt import XrtBackend

        backend = XrtBackend(
            {"backend": {"xrt": {"xclbin_path": xclbin_path, "target": target}}},
            persistent=True,
        )
        return cls(backend, base_params=base_params)

    def run(
        self,
        kernel_class: type[Kernel],
        inputs: dict[str, torch.Tensor | Tensor] | None = None,
        *,
        verify: bool = False,
        **params: Any,
    ) -> dict[str, Tensor]:
        """Execute a single kernel eagerly.

        Args:
            kernel_class: Kernel subclass to execute.
            inputs: Mapping of tensor name to input data.
                torch.Tensor → layout + serialize + LOAD + PUSH
                Tensor(on_device) → skip (BO already on device)
            verify: If True, compare HW output against behavioral model
                    golden (same CompositeKernel.forward() chain as vten run).
                    Golden data is stored on output Tensor._golden_data for
                    multi-layer chaining.
            **params: Kernel parameters (merged with base_params).

        Returns:
            Dict of output tensor names → Tensor.
            On HW backend: Tensor(on_device=True).
            On SIM backend: Tensor with .data set (host).

        Raises:
            VerificationError: If verify=True and HW output doesn't match golden.
        """
        inputs = inputs or {}
        merged = {**self._base_params, **params}
        spec = merged.pop("_spec", None)

        self._run_count += 1
        run_t0 = _time.monotonic()

        # Layer banner — show run number and key params for context
        label = params.get("name", "") or kernel_class.__name__
        n_device = sum(
            1 for d in inputs.values()
            if isinstance(d, Tensor) and d.on_device
        )
        n_host = len(inputs) - n_device
        input_desc = f"{n_host} host"
        if n_device:
            input_desc += f", {n_device} device"
        logger.info(
            "──── run #%d: %s (%s) ────",
            self._run_count, label, input_desc,
        )

        ctx = ExecutionContext(
            backend=self._backend,
            project_params=merged,
            mode="inference",
        )
        ki = ctx.instantiate(kernel_class, spec=spec, **merged)

        # Bind inputs
        for name, data in inputs.items():
            tensor = ki.get_tensor(name)
            if isinstance(data, Tensor) and data.on_device:
                # Device tensor → bind BO, skip LOAD+PUSH
                # When verify=True, skip prebound and serialize fresh
                # (eliminates upload/main-run BO discrepancy)
                if not verify:
                    ctx.bind_device_buffer(tensor, data._bo)
                # Set host-side data for two purposes:
                # 1. Stage 3 serialization (verify mode re-serializes fresh)
                # 2. Golden computation in _verify_outputs() (forward() reads tensor data)
                # Prefer golden (verified chain) > data (STORE readback) > zeros.
                chain_data = data.golden if data.golden is not None else data.data
                if chain_data is not None:
                    tensor.data = chain_data
                else:
                    shape = tensor._resolved_shape or tensor.shape
                    tensor.data = torch.zeros(shape, dtype=tensor.dtype)
            else:
                # Host tensor → assign data for normal LOAD+PUSH
                if isinstance(data, Tensor):
                    tensor.data = data.data
                else:
                    tensor.data = data

        # Execute kernel's DSL sequence
        ki.run(ctx)

        # Compile + execute
        result = ctx.run()

        # Log execution summary (phase-by-phase)
        self._log_execution_summary()

        # Output Tensor objects (with BO binding for HW backends)
        outputs = {name: t for name, t in result.output_tensors.items()}

        # Verify against behavioral model golden
        if verify:
            self._verify_outputs(ki.kernel_class_instance, inputs, outputs,
                                 compiled=ctx._last_compiled)

        run_elapsed = _time.monotonic() - run_t0
        logger.info("  total: %s", format_elapsed(run_elapsed))

        return outputs

    def _log_execution_summary(self) -> None:
        """Log execution phase summary from backend's interpreter."""
        summary = getattr(self._backend, "get_execution_summary", lambda: None)()
        if summary is None:
            return
        for p in summary.phases:
            if p.phase == "configure":
                logger.info("  configure: %d regs (%s)", p.n_cmds, format_elapsed(p.elapsed))
            elif p.phase == "send":
                logger.info("  send: %d tensors (%s)", p.n_tensors, format_size(p.n_bytes))
            elif p.phase == "poll":
                logger.info("  poll: %s, %d polls", format_elapsed(p.elapsed), p.n_polls)
            elif p.phase == "recv":
                logger.info("  recv: %d tensors (%s)", p.n_tensors, format_size(p.n_bytes))
            # trigger: skip (vsync detail unnecessary)

    def _verify_outputs(
        self,
        kernel_inst: object,
        inputs: dict[str, torch.Tensor | Tensor],
        outputs: dict[str, Tensor],
        compiled: object | None = None,
    ) -> None:
        """Verify HW outputs against golden using shared golden computation.

        Uses runtime.golden.compute_golden_outputs() — identical logic to
        CLI's _compute_auto_golden + _apply_unlayout path.

        Sub-kernel tensor data is already set from input binding in run().
        For device-resident inputs (chained outputs), the golden/data from
        the previous layer was set on the sub-kernel tensor during binding.
        """
        from vten.runtime.golden import compute_golden_outputs

        if compiled is None:
            return

        view = compiled.flattened_view

        # compute_golden_outputs uses the same forward() + format conversion
        # + unlayout pipeline as CLI verification
        golden_map = compute_golden_outputs(kernel_inst, view)

        for name, out_tensor in outputs.items():
            golden = golden_map.get(name)
            if golden is None:
                continue
            hw_logical = out_tensor.cpu()
            from vten.runtime.verifier import check_match
            check_match(name, hw_logical, golden)
            logger.info("verify: %s PASS shape=%s", name, tuple(hw_logical.shape))
            out_tensor.golden = golden

    def upload(
        self,
        data: torch.Tensor,
        tensor_name: str,
        kernel_class: type[Kernel],
        params: dict | None = None,
    ) -> Tensor:
        """Upload a tensor to device memory (1-time, for weights/biases).

        Args:
            data: Host tensor data (logical shape).
            tensor_name: Name of the tensor in the kernel class.
            kernel_class: Kernel class (for layout method lookup).
            params: Parameters for kernel instantiation (shape resolution).

        Returns:
            Tensor(on_device=True) with BO bound.
        """
        merged = {**self._base_params, **(params or {})}
        spec = merged.pop("_spec", None)

        logger.info(
            "upload: %s (%s, %s)",
            tensor_name, list(data.shape), format_size(data.numel() * data.element_size()),
        )

        # For CompositeKernel, find the sub-kernel that owns tensor_name
        # and instantiate only that sub-kernel (avoids serializing unrelated tensors).
        from vten.kernel.composite import CompositeKernel
        upload_cls = kernel_class
        if isinstance(kernel_class, type) and issubclass(kernel_class, CompositeKernel):
            for (sub_name, t_name), exposed_name in kernel_class._auto_exposed.items():
                if t_name == tensor_name or exposed_name == tensor_name:
                    upload_cls = kernel_class._sub_kernel_refs[sub_name]
                    break

        # Create a temporary context to instantiate kernel for layout/packing info
        ctx = ExecutionContext(
            backend=self._backend,
            project_params=merged,
            mode="inference",
        )
        ki = ctx.instantiate(upload_cls, spec=spec, **merged)
        tensor = ki.get_tensor(tensor_name)

        # Assign data and let the normal compile pipeline handle layout+serialize
        tensor.data = data

        # Use push_tensor to record LOAD+PUSH ops
        h = ctx.push_tensor(tensor)

        # Compile and execute to create the BO on device
        result = ctx.run()

        # Extract BO from interpreter
        compiled = ctx._last_compiled
        view = compiled.flattened_view
        exposed = view.exposed_tensors.get(tensor_name)

        t = Tensor(
            shape=tensor._resolved_shape or tensor.shape,
            dtype=tensor.dtype,
            interface=tensor.interface,
            direction=tensor.direction,
        )
        t.name = tensor_name
        t._resolved_shape = tensor._resolved_shape
        t._element_count = tensor._element_count

        # Store logical data for golden chain (verify mode)
        t._golden_data = data

        is_hw = self._backend.compile_target == "hw"

        if is_hw and exposed is not None:
            buffer_id = compiled.buffer_ids.get(tensor_name)
            if buffer_id is not None:
                bo = self._backend.get_buffer_object(buffer_id)
                if bo is not None:
                    bo_size = bo.size() if hasattr(bo, "size") else exposed._serialized_size
                    t._bind_bo(bo, bo_size)
        else:
            # SIM: store host data
            t.data = data

        return t

    def run_pipeline(
        self,
        kernel_class: type[Kernel],
        layers: list[dict],
        inputs: dict[str, torch.Tensor | Tensor],
        per_layer_inputs: list[dict[str, torch.Tensor | Tensor]] | None = None,
        chain: dict[str, str] | None = None,
        verify: bool = False,
    ) -> dict[str, Tensor]:
        """Sequential chain convenience. Internally calls run() per layer.

        Args:
            kernel_class: Kernel class for all layers.
            layers: Per-layer parameter dicts.
            inputs: Initial inputs (first layer).
            per_layer_inputs: Per-layer additional inputs (weights, biases).
            chain: Output→input name mapping (default: {"ofm_mem": "ifm_mem"}).
            verify: If True, verify each layer against behavioral model golden.

        Returns:
            Output dict from the last layer.
        """
        chain = chain or {"ofm_mem": "ifm_mem"}
        per_layer_inputs = per_layer_inputs or [{} for _ in layers]

        n = len(layers)
        logger.info("════ pipeline: %d layers (%s) ════", n, kernel_class.__name__)
        pipe_t0 = _time.monotonic()

        current = dict(inputs)
        result: dict[str, Tensor] | None = None
        for i, layer_params in enumerate(layers):
            layer_params = {**layer_params, "name": layer_params.get("name", f"layer {i}/{n}")}
            merged_inputs = {**current, **per_layer_inputs[i]}
            result = self.run(
                kernel_class, inputs=merged_inputs, verify=verify, **layer_params,
            )
            # Chain: map output names to next layer's input names
            current = {
                dst: result[src]
                for src, dst in chain.items()
                if src in result
            }

        if result is None:
            raise ValueError("layers list is empty")

        pipe_elapsed = _time.monotonic() - pipe_t0
        logger.info(
            "════ pipeline done: %d layers, %s ════", n, format_elapsed(pipe_elapsed),
        )
        return result

    def cleanup(self) -> None:
        """Release all device resources. BO references become invalid."""
        self._backend.cleanup()


class InferenceModule(torch.nn.Module):
    """nn.Module wrapper for FPGA kernel inference.

    Subclass and set kernel_cls, input_name, output_name.

    Usage::

        class NPUConv3D(InferenceModule):
            kernel_cls = NpuPipelineKernel
            input_name = "ifm_mem"
            output_name = "ofm_mem"

        conv = NPUConv3D(session, weight=w, bias=b, **params)
        y = conv(x)  # Tensor(on_device)
        y.cpu()       # → torch.Tensor

        # With verification:
        y = conv(x, verify=True)
    """

    kernel_cls: type[Kernel]
    input_name: str = "ifm_mem"
    output_name: str = "ofm_mem"

    def __init__(
        self,
        session: InferenceSession,
        *,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        weight_name: str = "wgt_mem",
        bias_name: str = "bias_mem",
        **params: Any,
    ) -> None:
        super().__init__()
        self._session = session
        self._params = params
        self._extra_inputs: dict[str, Tensor] = {}

        if weight is not None:
            self._extra_inputs[weight_name] = session.upload(
                weight, weight_name, self.kernel_cls, params,
            )
        if bias is not None:
            self._extra_inputs[bias_name] = session.upload(
                bias, bias_name, self.kernel_cls, params,
            )

    def forward(
        self,
        x: torch.Tensor | Tensor,
        *,
        verify: bool = False,
        **extra_inputs: torch.Tensor | Tensor,
    ) -> Tensor:
        """Execute kernel. Returns Tensor(on_device=True).

        Args:
            x: Primary input tensor.
            verify: If True, verify HW output against behavioral model golden.
            **extra_inputs: Additional per-call inputs (e.g. concat_mem=skip).
        """
        inputs: dict[str, torch.Tensor | Tensor] = {self.input_name: x}
        inputs.update(self._extra_inputs)
        inputs.update(extra_inputs)
        result = self._session.run(
            self.kernel_cls, inputs=inputs, verify=verify, **self._params,
        )
        return result[self.output_name]
