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
        """Run forward() on kernel instance and return serialized outputs.

        Flow:
          1. Find kernel instance from compiled.flattened_view
          2. Call run_forward() to compute golden outputs (physical format)
          3. Serialize each DEV_TO_HOST tensor to bytes
          4. Return BackendResult with output_buffers
        """
        from vten.runtime.golden import run_forward
        from vten.runtime.serializer import StreamSerializer
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

        # Serialize outputs into buffer bytes
        output_buffers: dict[int, bytes] = {}
        for name, exposed in view.exposed_tensors.items():
            if exposed.direction != Direction.DEV_TO_HOST:
                continue
            if name not in fwd_result:
                continue

            physical = fwd_result[name]

            try:
                iface = view.top_spec.get_interface(exposed.top_interface)
            except KeyError:
                continue
            if iface.packing is None:
                continue

            serializer = StreamSerializer(iface.packing)

            # Handle multi-port (array) tensors
            if exposed._port_buffers:
                raw = serializer.serialize(physical.flatten())
                n_ports = len(exposed._port_buffers)
                bytes_per_beat = (iface.packing.bus_width + 7) // 8
                total_beats = len(raw) // bytes_per_beat

                if n_ports > 1 and total_beats >= n_ports:
                    beats_per_port = total_beats // n_ports
                    for idx, port_name in enumerate(exposed._port_buffers):
                        key = f"{name}:{port_name}"
                        bid = compiled.buffer_ids.get(key)
                        if bid is not None:
                            start = idx * beats_per_port * bytes_per_beat
                            end = start + beats_per_port * bytes_per_beat
                            output_buffers[bid] = raw[start:end]
                else:
                    for port_name in exposed._port_buffers:
                        key = f"{name}:{port_name}"
                        bid = compiled.buffer_ids.get(key)
                        if bid is not None:
                            output_buffers[bid] = raw
            else:
                raw = serializer.serialize(physical.flatten())
                bid = compiled.buffer_ids.get(name)
                if bid is not None:
                    output_buffers[bid] = raw

        return BackendResult(
            status=0,
            output_buffers=output_buffers,
        )

    def cleanup(self) -> None:
        pass

    @property
    def compile_target(self) -> CompileTarget:
        return CompileTarget.SIM
