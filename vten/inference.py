"""Inference API — Kernel-Granular Eager Execution.

Provides InferenceSession (eager executor) and InferenceModule (nn.Module wrapper)
for running verified kernels on real FPGA hardware.

Spec reference: 11_inference_api.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from vten.kernel.tensor import Tensor
from vten.runtime.context import ExecutionContext
from vten.spec.models import Direction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vten.backend.base import Backend
    from vten.kernel.base import Kernel


class VerificationError(Exception):
    """Raised when HW output doesn't match behavioral model golden."""
    pass


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
    ) -> None:
        """Create an inference session.

        Args:
            backend: Backend instance, or backend name ("xrt") for auto-setup.
            base_params: Default parameters for all run() calls.
            kernel: Kernel name for xclbin auto-discovery (e.g. "npu_pipeline").
            target: "hw" or "hw_emu" (used with string backend).
            project_dir: Project root containing vten.toml (default: ".").
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
                ctx.bind_device_buffer(tensor, data._bo)
                # Assign zero data so forward() (called inside verify()) won't crash
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

        # Wrap outputs
        outputs = self._wrap_outputs(result, ctx, ki)

        # Verify against behavioral model golden
        if verify:
            self._verify_golden(kernel_class, merged, inputs, outputs)

        return outputs

    def _verify_golden(
        self,
        kernel_class: type[Kernel],
        params: dict,
        inputs: dict[str, torch.Tensor | Tensor],
        outputs: dict[str, Tensor],
    ) -> None:
        """Compute golden via behavioral model and compare with HW outputs.

        Uses the same CompositeKernel.forward() chain as vten run:
          1. Create fresh kernel instance with resolved params
          2. Set golden input data on sub-kernel tensors (logical format)
          3. forward() applies layout → multi-round dataflow → physical output
          4. unlayout → logical output
          5. Compare with HW output (.cpu())
          6. Store golden on output Tensor._golden_data for chain
        """
        from vten.kernel.composite import CompositeKernel

        # Create a fresh kernel instance for golden computation (no backend)
        golden_ctx = ExecutionContext(project_params=params)
        golden_ki = golden_ctx.instantiate(kernel_class, **params)
        golden_kernel = golden_ki.kernel_class_instance

        if isinstance(golden_kernel, CompositeKernel):
            self._verify_composite(golden_kernel, inputs, outputs)
        else:
            self._verify_simple(golden_kernel, inputs, outputs)

    def _verify_composite(
        self,
        golden_kernel: Any,
        inputs: dict[str, torch.Tensor | Tensor],
        outputs: dict[str, Tensor],
    ) -> None:
        """Golden verification for CompositeKernel."""
        cls = type(golden_kernel)

        # Build reverse mapping: exposed_name → (sub_name, tensor_name)
        reverse: dict[str, tuple[str, str]] = {}
        for (sub_name, tensor_name), exposed_name in cls._auto_exposed.items():
            reverse[exposed_name] = (sub_name, tensor_name)
            if tensor_name not in reverse:
                reverse[tensor_name] = (sub_name, tensor_name)

        # Set golden input data on sub-kernel tensors (logical format).
        # CompositeKernel.forward() reads tensor.data and applies layout_*().
        for input_name, data in inputs.items():
            if input_name not in reverse:
                continue
            sub_name, tensor_name = reverse[input_name]
            sub = golden_kernel._get_sub_kernel_instance(sub_name)
            if sub is None:
                continue
            t = sub.get_tensor(tensor_name)

            # Resolve golden data: _golden_data → .data → raw torch.Tensor
            if isinstance(data, Tensor):
                golden_data = data._golden_data if data._golden_data is not None else data.data
            elif isinstance(data, torch.Tensor):
                golden_data = data
            else:
                continue

            if golden_data is not None:
                t.data = golden_data

        # Run behavioral model (same chain as vten run golden)
        golden_result = golden_kernel.forward()

        # Compare each output
        for name, out_tensor in outputs.items():
            if name not in golden_result:
                continue

            golden_phys = golden_result[name].flatten()

            # Apply unlayout if available on the owning sub-kernel
            golden_logical: torch.Tensor
            if name in reverse:
                sub_name, tensor_name = reverse[name]
                sub = golden_kernel._get_sub_kernel_instance(sub_name)
                unlayout_fn = getattr(sub, f"unlayout_{tensor_name}", None) if sub else None
                if unlayout_fn is not None:
                    golden_logical = unlayout_fn(golden_phys)
                else:
                    golden_logical = golden_phys
            else:
                golden_logical = golden_phys

            hw_logical = out_tensor.cpu()

            if not torch.equal(hw_logical, golden_logical):
                n_diff = int((hw_logical != golden_logical).sum().item())
                n_total = hw_logical.numel()
                max_diff = int((hw_logical.int() - golden_logical.int()).abs().max().item())
                raise VerificationError(
                    f"Tensor '{name}' verify FAIL: "
                    f"{n_diff}/{n_total} elements differ, "
                    f"max_diff={max_diff}, "
                    f"hw_shape={tuple(hw_logical.shape)}, "
                    f"golden_shape={tuple(golden_logical.shape)}"
                )
            logger.info("verify: %s PASS shape=%s", name, tuple(hw_logical.shape))

            # Store golden for multi-layer chain
            out_tensor._golden_data = golden_logical

    def _verify_simple(
        self,
        golden_kernel: Any,
        inputs: dict[str, torch.Tensor | Tensor],
        outputs: dict[str, Tensor],
    ) -> None:
        """Golden verification for simple (non-composite) kernel."""
        # Collect H2D inputs with layout applied
        fwd_inputs: dict[str, torch.Tensor] = {}
        for t in golden_kernel.tensors():
            if t.data is None:
                continue
            direction = getattr(t, "direction", None)
            if direction is None or direction.value == "host_to_dev":
                layout_fn = getattr(golden_kernel, f"layout_{t.name}", None)
                if layout_fn is not None and callable(layout_fn):
                    fwd_inputs[t.name] = layout_fn(t.data)
                else:
                    fwd_inputs[t.name] = t.data

        golden_result = golden_kernel.forward(**fwd_inputs)

        for name, out_tensor in outputs.items():
            if name not in golden_result:
                continue

            golden_phys = golden_result[name].flatten()
            unlayout_fn = getattr(golden_kernel, f"unlayout_{name}", None)
            if unlayout_fn is not None:
                golden_logical = unlayout_fn(golden_phys)
            else:
                golden_logical = golden_phys

            hw_logical = out_tensor.cpu()

            if not torch.equal(hw_logical, golden_logical):
                n_diff = int((hw_logical != golden_logical).sum().item())
                n_total = hw_logical.numel()
                max_diff = int((hw_logical.int() - golden_logical.int()).abs().max().item())
                raise VerificationError(
                    f"Tensor '{name}' verify FAIL: "
                    f"{n_diff}/{n_total} elements differ, max_diff={max_diff}"
                )
            logger.info("verify: %s PASS shape=%s", name, tuple(hw_logical.shape))
            out_tensor._golden_data = golden_logical

    def _wrap_outputs(
        self,
        result: Any,
        ctx: ExecutionContext,
        ki: Any,
    ) -> dict[str, Tensor]:
        """Wrap kernel outputs as Tensor objects.

        HW backend: Tensor(on_device=True) with BO bound.
        SIM backend: Tensor with .data = deserialized torch.Tensor.
        """
        compiled = ctx._last_compiled
        if compiled is None:
            return {}

        is_hw = getattr(self._backend, "compile_target", "sim") == "hw"
        interpreter = getattr(self._backend, "_interpreter", None)
        view = compiled.flattened_view
        outputs: dict[str, Tensor] = {}

        for name, exposed in view.exposed_tensors.items():
            if exposed.direction != Direction.DEV_TO_HOST:
                continue

            origin = exposed.origin_tensor
            t = Tensor(
                shape=origin._resolved_shape or origin.shape,
                dtype=origin.dtype,
                interface=origin.interface,
                direction=Direction.DEV_TO_HOST,
            )
            t.name = name
            t._resolved_shape = origin._resolved_shape
            t._element_count = origin._element_count

            if is_hw and interpreter is not None:
                # HW: bind BO from interpreter → Tensor(on_device=True)
                buffer_id = compiled.buffer_ids.get(name)
                if buffer_id is not None:
                    bo = interpreter._buffers.get(buffer_id)
                    if bo is not None:
                        deserialize_fn = self._make_deserialize_fn(
                            view, exposed,
                        )
                        t._bind_bo(bo, exposed._serialized_size, deserialize_fn)
                        outputs[name] = t
                        continue

            # SIM or fallback: output is already deserialized by ctx.run()
            if name in result.output_tensors:
                t.data = result.output_tensors[name]

            outputs[name] = t

        return outputs

    def _make_deserialize_fn(
        self,
        view: Any,
        exposed: Any,
    ) -> Any:
        """Create a bytes → torch.Tensor deserialize function for .cpu().

        Includes packing deserialization and unlayout if applicable.
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

        # Create a temporary context to instantiate kernel for layout/packing info
        ctx = ExecutionContext(
            backend=self._backend,
            project_params=merged,
            mode="inference",
        )
        ki = ctx.instantiate(kernel_class, spec=spec, **merged)
        tensor = ki.get_tensor(tensor_name)

        # Assign data and let the normal compile pipeline handle layout+serialize
        tensor.data = data

        # Use send_tensor to record LOAD+PUSH ops
        h = ctx.send_tensor(tensor)

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

        is_hw = getattr(self._backend, "compile_target", "sim") == "hw"
        interpreter = getattr(self._backend, "_interpreter", None)

        if is_hw and interpreter is not None and exposed is not None:
            buffer_id = compiled.buffer_ids.get(tensor_name)
            if buffer_id is not None:
                bo = interpreter._buffers.get(buffer_id)
                if bo is not None:
                    t._bind_bo(bo, exposed._serialized_size)
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

        current = dict(inputs)
        result: dict[str, Tensor] | None = None
        for i, layer_params in enumerate(layers):
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
