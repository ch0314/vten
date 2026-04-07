"""Probe management — declarative probe annotation and golden extraction.

Extracted from ExecutionContext. Functions take state as parameters
to stay decoupled from ExecutionContext internals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vten.spec.models import OpKind

if TYPE_CHECKING:
    from vten.dsl.operations import Operation, OperationHandle


def apply_declarative_probes(
    probes: list[str],
    pending_ops: list[Operation],
) -> list[tuple[str, str]]:
    """Post-hoc annotation: mark ops as probes based on declarative specs.

    For output probes (simple name like "data_out"): set probe=True on
    matching PULL/RECV operations.
    For internal probes (dotted name like "scale.data_out"): return as
    (sub_name, tensor_name) tuples.

    Returns:
        List of internal probe requests (sub_name, tensor_name).
    """
    internal_requests: list[tuple[str, str]] = []
    for probe_spec in probes:
        if "." in probe_spec:
            sub, tensor = probe_spec.rsplit(".", 1)
            internal_requests.append((sub, tensor))
        else:
            for op in pending_ops:
                if (
                    op.kind in (OpKind.PULL_TENSOR, OpKind.RECV_TENSOR)
                    and op.tensor is not None
                    and op.tensor.name == probe_spec
                ):
                    op.probe = True
    return internal_requests


def resolve_internal_probe_golden(
    internal_probe_requests: list[tuple[str, str]],
    kernels: dict,
    internal_probe_golden: dict[tuple[str, str], torch.Tensor],
) -> None:
    """Auto-extract internal probe golden from CompositeKernel forward results.

    Mutates internal_probe_golden in place.
    """
    if not internal_probe_requests:
        return
    for ki in kernels.values():
        inst = ki.kernel_class_instance
        pool = getattr(inst, "_golden_pool", None)
        if pool is None:
            continue
        for sub_name, tensor_name in internal_probe_requests:
            if (sub_name, tensor_name) in internal_probe_golden:
                continue
            key = (sub_name, tensor_name)
            if key in pool:
                internal_probe_golden[key] = pool[key]


def collect_probe_golden_tensors(
    pending_ops: list[Operation],
    verifications: list,
    compute_auto_golden_fn,
) -> dict[str, torch.Tensor]:
    """Collect golden tensors for probe-enabled PULL operations.

    Args:
        pending_ops: Current batch of operations.
        verifications: List of VerificationTask objects.
        compute_auto_golden_fn: Callable(op_handle) → torch.Tensor.

    Returns:
        tensor_name → golden torch.Tensor.
    """
    probe_tensor_names: set[str] = set()
    for op in pending_ops:
        if op.probe and op.tensor is not None:
            probe_tensor_names.add(op.tensor.name)

    probe_golden: dict[str, torch.Tensor] = {}
    for task in verifications:
        op = task.op_handle.op
        if op.tensor is None:
            continue
        tensor_name = op.tensor.name
        if tensor_name in probe_tensor_names and tensor_name not in probe_golden:
            golden = task.golden
            if golden is None:
                golden = compute_auto_golden_fn(task.op_handle)
            probe_golden[tensor_name] = golden
    return probe_golden
