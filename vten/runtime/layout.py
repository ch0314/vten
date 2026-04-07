"""Layout helpers — apply/unlayout transformations for tensor serialization.

Extracted from RuntimeEngine. Pure static functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vten.runtime.flattener import ExposedTensor, FlattenedKernelView


def apply_layout(
    view: FlattenedKernelView,
    exposed: ExposedTensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Apply layout_{name}() if the owning kernel defines it.

    Returns physical data for serialization, or the original data if
    no layout method exists.
    """
    sub_name = exposed.origin_path.split(".")[0]
    ki = view.sub_kernels.get(sub_name)
    if ki is None or ki.kernel_class_instance is None:
        return data
    tensor_name = exposed.origin_tensor.name
    layout_fn = getattr(ki.kernel_class_instance, f"layout_{tensor_name}", None)
    if layout_fn is not None and callable(layout_fn):
        return layout_fn(data)
    return data


def apply_unlayout(
    view: FlattenedKernelView,
    exposed: ExposedTensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """Apply unlayout_{name}() if the owning kernel defines it.

    Returns logical data for user output, or the original data if
    no unlayout method exists.
    """
    sub_name = exposed.origin_path.split(".")[0]
    ki = view.sub_kernels.get(sub_name)
    if ki is None or ki.kernel_class_instance is None:
        return data
    tensor_name = exposed.origin_tensor.name
    unlayout_fn = getattr(ki.kernel_class_instance, f"unlayout_{tensor_name}", None)
    if unlayout_fn is not None and callable(unlayout_fn):
        return unlayout_fn(data)
    return data
