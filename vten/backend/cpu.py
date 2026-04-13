"""CPU backend — runs kernel forward() as a reference model.

No RTL simulation, no FPGA. Executes the kernel's Python forward()
computation and returns outputs as if they were DUT results.

Use cases:
  - Quick config/pipeline smoke test without HW
  - Golden reference data generation
  - Kernel forward() debugging

Usage:
  vten run --backend cpu --kernel fmapIO --verify
  (verify will always PASS since DUT output == golden output)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vten.backend.base import Backend, BackendResult, CompileTarget

if TYPE_CHECKING:
    from vten.runtime.engine import CompiledResult

logger = logging.getLogger(__name__)


class CpuBackend(Backend):
    """Execute kernel forward() in pure Python/PyTorch."""

    def __init__(self, config: dict | None = None, **kwargs) -> None:
        from vten.backend.base import RunContext

        self._config = config or {}
        self._run_ctx = RunContext()

    def execute(self, compiled: CompiledResult) -> BackendResult:
        """Run forward() on kernel instance and return outputs.

        Fast path (default): stores forward() tensor results directly
        in BackendResult._forward_tensors, skipping serialize/deserialize.
        This makes CPU backend ~100x faster for large tensors.

        The read_output_tensors() in output_reader.py detects _forward_tensors
        and uses them directly instead of going through byte serialization.
        """
        from vten.runtime.golden import run_forward
        from vten.spec.models import Direction

        view = compiled.flattened_view

        # Use top-level kernel instance (Composite or Simple)
        kernel_inst = compiled.kernel_instance
        if kernel_inst is None:
            # Fallback for backward compat: first sub-kernel
            for ki in view.sub_kernels.values():
                if ki.kernel_class_instance is not None:
                    kernel_inst = ki.kernel_class_instance
                    break

        if kernel_inst is None:
            logger.warning("cpu backend: no kernel instance found, returning empty")
            return BackendResult(status=0)

        # Run forward() → physical output tensors
        fwd_result = run_forward(kernel_inst)
        logger.info("cpu forward: %s", list(fwd_result.keys()))

        # Collect forward tensors for D2H outputs (fast path: no serialization)
        forward_tensors: dict[str, object] = {}
        for name, exposed in view.exposed_tensors.items():
            if exposed.direction != Direction.DEV_TO_HOST:
                continue
            if name in fwd_result:
                forward_tensors[name] = fwd_result[name]

        return BackendResult(
            status=0,
            _forward_tensors=forward_tensors if forward_tensors else None,
        )

    def cleanup(self) -> None:
        pass

    @property
    def compile_target(self) -> CompileTarget:
        return CompileTarget.SIM
