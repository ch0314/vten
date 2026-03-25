"""High-level functional API for torch tensor in/out execution.

Provides run_kernel() for one-shot calls and KernelExecutor for
repeated calls with automatic cross-batch alias.

Usage:
    # One-shot
    outputs = run_kernel(MyKernel, {"ifm": x}, backend=b, params={...})

    # Repeated with auto-alias
    npu = KernelExecutor(MyKernel, backend=b)
    y = npu(ifm=x, wgt=w)["ofm"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vten.runtime.context import ExecutionContext

if TYPE_CHECKING:
    from vten.backend.base import Backend
    from vten.kernel.base import Kernel


def run_kernel(
    kernel_class: type[Kernel],
    inputs: dict[str, torch.Tensor],
    *,
    backend: Backend | None = None,
    params: dict | None = None,
    configure: bool = False,
) -> dict[str, torch.Tensor]:
    """Execute a kernel once with given input tensors, return output tensors.

    Automatically generates DSL operations:
      H2D tensors (in inputs) → send_tensor (= LOAD + PUSH)
      configure=True          → configure (= WRITE_REG × N)
      D2H tensors (not in inputs) → recv_tensor (= PULL + STORE)

    Args:
        kernel_class: Kernel subclass to execute.
        inputs: Mapping of tensor name → torch.Tensor for HOST_TO_DEV tensors.
        backend: Backend instance (XsimBackend, XrtBackend, etc.).
        params: Runtime parameters forwarded to kernel instantiation.
        configure: If True, emit configure() op for auto_bind registers.

    Returns:
        Dict mapping output tensor names to deserialized torch.Tensor results.
    """
    params = params or {}
    ctx = ExecutionContext(backend=backend, project_params=params)
    ki = ctx.instantiate(kernel_class, **params)

    # Assign input tensor data
    for name, data in inputs.items():
        ki.get_tensor(name).data = data

    # Auto-generate DSL ops
    send_handles = []
    for t in ki.tensors():
        if t.name in inputs:
            h = ctx.send_tensor(t)
            send_handles.append(h)

    if configure:
        dep = send_handles[0] if send_handles else None
        ctx.configure(ki, dep=dep)

    for t in ki.tensors():
        if t.name not in inputs:
            dep = send_handles[-1] if send_handles else None
            ctx.recv_tensor(t, dep=dep)

    result = ctx.run()
    return result.output_tensors


class KernelExecutor:
    """Reusable kernel executor with automatic cross-batch alias.

    When an output tensor from a previous call is passed as input to
    the next call, the executor automatically applies alias to reuse
    the SHM buffer (LOAD skip, no redundant host→device transfer).

    Usage:
        npu = KernelExecutor(NPU3DKernel, backend=backend)
        x = input_tensor
        for i, p in enumerate(layers):
            x = npu(ifm=x, wgt=weights[i], bias=biases[i], **p)["ofm"]
    """

    def __init__(
        self,
        kernel_class: type[Kernel],
        backend: Backend | None = None,
        params: dict | None = None,
        configure: bool = False,
    ) -> None:
        self._kernel_class = kernel_class
        self._backend = backend
        self._base_params = params or {}
        self._configure = configure
        # Track previous call's output tensors for auto-alias
        self._prev_outputs: dict[int, str] = {}  # id(tensor) → tensor_name
        self._prev_ki: object | None = None
        self._prev_ctx: ExecutionContext | None = None

    def __call__(
        self, *, _params: dict | None = None, **inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Execute kernel with given input tensors, return output tensors.

        Args:
            _params: Per-call parameter overrides (merged with base params).
            **inputs: name=torch.Tensor pairs for HOST_TO_DEV tensors.

        Returns:
            Dict mapping output tensor names to torch.Tensor results.
        """
        params = {**self._base_params, **(_params or {})}
        ctx = ExecutionContext(backend=self._backend, project_params=params)
        ki = ctx.instantiate(self._kernel_class, **params)

        # Assign input data + detect auto-alias opportunities
        alias_applied = set()
        for name, data in inputs.items():
            tensor = ki.get_tensor(name)
            tensor_id = id(data)
            if (
                self._prev_ki is not None
                and tensor_id in self._prev_outputs
            ):
                # This input was a previous output → alias for buffer reuse
                prev_name = self._prev_outputs[tensor_id]
                prev_tensor = self._prev_ki.get_tensor(prev_name)
                ctx.alias(prev_tensor, tensor)
                alias_applied.add(name)
            else:
                tensor.data = data

        # Auto-generate DSL ops
        send_handles = []
        for t in ki.tensors():
            if t.name in inputs:
                h = ctx.send_tensor(t)
                send_handles.append(h)

        if self._configure:
            dep = send_handles[0] if send_handles else None
            ctx.configure(ki, dep=dep)

        for t in ki.tensors():
            if t.name not in inputs:
                dep = send_handles[-1] if send_handles else None
                ctx.recv_tensor(t, dep=dep)

        result = ctx.run()

        # Track outputs for next call's auto-alias
        self._prev_outputs = {
            id(tensor): name
            for name, tensor in result.output_tensors.items()
        }
        self._prev_ki = ki
        self._prev_ctx = ctx

        return result.output_tensors
