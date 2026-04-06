"""DSL operation types and handles.

Spec reference: 00_data_models.md §1.5, §7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vten.spec.models import OpKind

if TYPE_CHECKING:
    from vten.kernel.tensor import Tensor

__all__ = ["OpKind", "Operation", "OperationHandle"]


@dataclass
class Operation:
    """Single DSL invocation record. Created during ExecutionContext recording."""

    kind: OpKind
    tensor: Tensor | None = None
    kernel: object | None = None  # KernelInstance (avoid circular import)
    register_interface: str | None = None
    register_fields: dict | None = None
    register_field_name: str | None = None  # For poll_register
    poll_expected: int | None = None  # Override expected value (None = all 1s)
    dep: list[OperationHandle] = field(default_factory=list)
    commit_dep: list[OperationHandle] = field(default_factory=list)
    probe: bool = False
    sync: bool = False
    golden: torch.Tensor | None = None
    verify: bool = False
    chunk_index: int | None = None
    chunk_total: int | None = None
    chunks_spec: int | list[int] | None = None  # original chunks arg
    config_group: int = 0  # multi-config group index (set by config_boundary)
    # Inference mode hints (used by IR lowering)
    _skip_data: bool = False  # send_tensor: BO already on device, skip LOAD+PUSH data


@dataclass
class OperationHandle:
    """Lightweight wrapper returned to user. Supports retroactive dependency."""

    op: Operation

    def add_commit_dependency(self, other: OperationHandle) -> None:
        """Delay this operation's commit until other's commit."""
        self.op.commit_dep.append(other)
