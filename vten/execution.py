"""Shared kernel execution core.

Provides ``execute_batch``, the atomic execution primitive shared by
``vten run`` (CLI) and ``InferenceSession`` (Python API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)


# ── Result types ──


@dataclass
class ConfigResult:
    """Result of a single config execution within a batch."""

    config_index: int
    result: Any | None = None  # ExecutionResult, None if errored
    error: Exception | None = None

    @property
    def passed(self) -> bool:
        return self.error is None

    @property
    def cycles(self) -> int:
        if self.result is None:
            return 0
        return self.result.total_cycles


@dataclass
class BatchResult:
    """Result of ``execute_batch``."""

    configs: list[ConfigResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.configs)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.configs if c.passed)

    @property
    def total_cycles(self) -> int:
        return max((c.cycles for c in self.configs), default=0)

    def single(self) -> Any:
        """Convenience accessor for single-config batches.

        Returns the ``ExecutionResult`` when the batch has exactly one
        config.  Raises if the batch is empty or multi-config.
        """
        if len(self.configs) != 1:
            raise ValueError(
                f"single() requires exactly 1 config, got {len(self.configs)}"
            )
        cr = self.configs[0]
        if cr.error is not None:
            raise cr.error
        return cr.result


# ── Core execution ──


def execute_batch(
    *,
    backend: Any,
    kernel_class: type,
    configs: list[dict],
    spec: Any = None,
    inputs: dict[str, Any] | None = None,
    verify: bool = False,
    lsb_tolerance: int | dict[str, int] = 0,
    project_dir: Path | None = None,
    probes: list[str] | None = None,
    seed: int = 42,
    quiet: bool = False,
    on_error: Literal["continue", "raise"] = "continue",
) -> BatchResult:
    """Execute a kernel with one or more configs.

    This is the shared primitive for ``vten run`` and
    ``InferenceSession``.  Each config gets its own
    :class:`~vten.runtime.context.ExecutionContext`, kernel
    instantiation, and compile+execute cycle.

    Args:
        backend: Backend instance.
        kernel_class: Kernel subclass to instantiate.
        configs: List of per-execution config dicts.  Each dict
            becomes both ``project_params`` (Tier 1) and
            ``runtime_params`` (Tier 5) for backward compatibility.
        spec: Optional KernelSpec override.
        inputs: Input tensor data (shared across all configs).
            Keys are tensor names.  Values: ``torch.Tensor`` (host)
            or ``vten.Tensor`` (possibly device-resident).
            When *None*, calls ``generate_inputs(seed=...)`` per config.
        verify: Compare HW output against ``forward()`` golden.
        lsb_tolerance: Opt-in integer-LSB tolerance for verification — an
            int applied to all outputs, or a dict tensor-name → int
            (missing names stay bit-exact). Default 0 keeps integer
            comparison bit-exact.
        project_dir: Project root for spec path resolution.
        probes: Declarative probe specifications.
        seed: Default RNG seed for ``generate_inputs()``.
            Each config can override via ``cfg.get("seed")``.
        quiet: Suppress compile-time logging (inference mode).
        on_error: ``"continue"`` catches errors and proceeds to next
            config; ``"raise"`` re-raises immediately.

    Returns:
        :class:`BatchResult` with per-config results.
    """
    from vten.runtime.context import ExecutionContext

    batch = BatchResult()

    for idx, cfg in enumerate(configs):
        try:
            ctx = ExecutionContext(
                backend=backend,
                project_params=cfg,
                project_dir=project_dir,
                mode="inference" if quiet else "verification",
            )

            ki = ctx.instantiate(kernel_class, spec=spec, **cfg)
            inst = ki.kernel_class_instance

            # Input assignment: user-provided data or auto-generated
            if inputs is not None:
                _assign_inputs(ctx, ki, inputs, verify=verify)
            else:
                config_seed = cfg.get("seed", seed)
                inst.generate_inputs(seed=config_seed)

            # Declarative probes
            if probes:
                ctx._register_declarative_probes(probes)

            # Execute kernel's DSL protocol (records ops on ctx)
            inst.run(ctx)

            if not ctx._pending_ops:
                # Scenario recorded no ops — count as pass
                from vten.runtime.context import ExecutionResult

                batch.configs.append(ConfigResult(
                    config_index=idx,
                    result=ExecutionResult(status="PASS"),
                ))
                continue

            # Compile → execute → verify
            result = ctx.run(verify=verify, lsb_tolerance=lsb_tolerance)
            batch.configs.append(ConfigResult(
                config_index=idx,
                result=result,
            ))

        except Exception as exc:
            if on_error == "raise":
                raise
            batch.configs.append(ConfigResult(
                config_index=idx,
                error=exc,
            ))
            logger.debug(
                "config %d/%d failed: %s", idx + 1, len(configs), exc,
                exc_info=True,
            )

    return batch


def _assign_inputs(
    ctx: Any,
    ki: Any,
    inputs: dict[str, Any],
    *,
    verify: bool = False,
) -> None:
    """Assign user-provided data to kernel tensors.

    Handles three input forms:

    - ``torch.Tensor`` — host data, assigned directly.
    - ``vten.Tensor(on_device=False)`` — extracts ``.data``.
    - ``vten.Tensor(on_device=True)`` — binds BO for zero-copy
      (skipped when *verify=True* to force re-serialization).
      Uses golden-chain data for ``forward()`` golden computation.
    """
    from vten.kernel.tensor import Tensor

    for name, data in inputs.items():
        tensor = ki.get_tensor(name)

        if isinstance(data, Tensor) and data.on_device:
            # Device-resident: bind BO for zero-copy, unless verifying
            if not verify:
                ctx.bind_device_buffer(tensor, data._bo)
            # Golden-chain data for forward() computation
            chain_data = data.golden if data.golden is not None else data.data
            if chain_data is not None:
                tensor.data = chain_data
            else:
                shape = tensor._resolved_shape or tensor.shape
                tensor.data = torch.zeros(shape, dtype=tensor.dtype)
        elif isinstance(data, Tensor):
            tensor.data = data.data
        else:
            tensor.data = data
