"""Dependency model utilities.

Spec reference: 01_kernel_and_dsl.md §3.2
"""

from __future__ import annotations

from vten.dsl.operations import OperationHandle


def normalize_deps(
    dep: OperationHandle | list[OperationHandle] | None,
) -> list[OperationHandle]:
    """Normalize dep parameter to a list of OperationHandles."""
    if dep is None:
        return []
    if isinstance(dep, OperationHandle):
        return [dep]
    return list(dep)
