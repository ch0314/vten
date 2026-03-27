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

import logging
from typing import TYPE_CHECKING

import torch

from vten.runtime.context import ExecutionContext

logger = logging.getLogger(__name__)

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
    spec: object | None = None,
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
        spec: Optional KernelSpec instance. If not provided, loaded from
              kernel_class.spec path or auto-generated.

    Returns:
        Dict mapping output tensor names to deserialized torch.Tensor results.
    """
    params = params or {}
    logger.info("run_kernel: %s, inputs=%s", kernel_class.__name__, list(inputs.keys()))
    ctx = ExecutionContext(backend=backend, project_params=params)
    ki = ctx.instantiate(kernel_class, spec=spec, **params)

    # Assign input tensor data
    for name, data in inputs.items():
        ki.get_tensor(name).data = data

    # Auto-generate DSL ops.
    # IMPORTANT: PULL must depend on LOAD (not PUSH) so that PUSH and PULL
    # can run concurrently — stream passthrough DUTs need both BFMs active.
    load_handles = []
    for t in ki.tensors():
        if t.name in inputs:
            h_load = ctx.load_tensor(t)
            load_handles.append(h_load)
            ctx.push_tensor(t, dep=h_load)

    if configure:
        dep = load_handles[0] if load_handles else None
        ctx.configure(ki, dep=dep)

    for t in ki.tensors():
        if t.name not in inputs:
            dep = load_handles[-1] if load_handles else None
            h_pull = ctx.pull_tensor(t, dep=dep)
            ctx.store_tensor(t, dep=h_pull)

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
        spec: object | None = None,
    ) -> None:
        self._kernel_class = kernel_class
        self._backend = backend
        self._base_params = params or {}
        self._configure = configure
        self._spec = spec
        # Track previous call's output tensors for auto-alias
        self._prev_outputs: dict[int, str] = {}  # id(tensor) → tensor_name
        self._prev_ki: object | None = None
        self._prev_ctx: ExecutionContext | None = None
        # Session state: managed here, not per-context
        self._session_open: bool = False

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
        # Transfer session state from executor to context
        ctx._session_open = self._session_open
        ki = ctx.instantiate(self._kernel_class, spec=self._spec, **params)

        # Assign input data + detect auto-alias opportunities
        logger.debug("KernelExecutor.__call__: %s, inputs=%s",
                      self._kernel_class.__name__, list(inputs.keys()))
        alias_applied = set()
        # Sim backends don't benefit from alias (SHM memcpy is cheap),
        # and submit_batch() overwrites the data region, breaking alias.
        # Only disable for actual sim backends; None backend (dry-run) is fine.
        can_alias = True
        if self._backend is not None:
            can_alias = getattr(self._backend, "compile_target", "sim") != "sim"
        for name, data in inputs.items():
            tensor = ki.get_tensor(name)
            tensor_id = id(data)
            if (
                can_alias
                and self._prev_ki is not None
                and tensor_id in self._prev_outputs
            ):
                # This input was a previous output → alias for buffer reuse
                prev_name = self._prev_outputs[tensor_id]
                prev_tensor = self._prev_ki.get_tensor(prev_name)
                ctx.alias(prev_tensor, tensor)
                alias_applied.add(name)
            else:
                tensor.data = data

        # Auto-generate DSL ops.
        # IMPORTANT: PULL must depend on LOAD (not PUSH) so that PUSH and PULL
        # can run concurrently — stream passthrough DUTs need both BFMs active.
        load_handles = []
        for t in ki.tensors():
            if t.name in inputs:
                h_load = ctx.load_tensor(t)
                load_handles.append(h_load)
                ctx.push_tensor(t, dep=h_load)

        if self._configure:
            dep = load_handles[0] if load_handles else None
            ctx.configure(ki, dep=dep)

        for t in ki.tensors():
            if t.name not in inputs:
                dep = load_handles[-1] if load_handles else None
                h_pull = ctx.pull_tensor(t, dep=dep)
                ctx.store_tensor(t, dep=h_pull)

        result = ctx.run()
        # Capture session state back from context
        self._session_open = ctx._session_open

        if alias_applied:
            logger.debug("auto-alias applied: %s", alias_applied)

        # Track outputs for next call's auto-alias
        self._prev_outputs = {
            id(tensor): name
            for name, tensor in result.output_tensors.items()
        }
        self._prev_ki = ki
        self._prev_ctx = ctx

        return result.output_tensors

    def close(self) -> None:
        """Close the backend session if one is open. Idempotent."""
        if self._session_open and self._backend is not None:
            if hasattr(self._backend, "close_session"):
                self._backend.close_session()
            self._session_open = False

    def __enter__(self) -> KernelExecutor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
