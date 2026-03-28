"""Stage 5: auto_bind & Bank Offset Resolution.

Spec reference: 02_runtime_engine.md §11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vten.errors import BindingError
from vten.spec.models import AutoBindSpec

if TYPE_CHECKING:
    from vten.runtime.flattener import FlattenedKernelView


@dataclass
class RegisterBindingEntry:
    """Result of auto_bind resolution."""

    register_name: str
    kernel_path: str
    interface_name: str
    absolute_offset: int
    auto_bind: AutoBindSpec
    resolved_value: int


def parse_bit_range(bit_range_str: str) -> tuple[int, int]:
    """Parse "hi:lo" → (hi, lo). Validates hi >= lo."""
    parts = bit_range_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid bit range '{bit_range_str}'. Expected 'hi:lo'."
        )
    hi, lo = int(parts[0]), int(parts[1])
    if hi < lo:
        raise ValueError(
            f"Invalid bit range '{bit_range_str}': hi ({hi}) < lo ({lo})."
        )
    return hi, lo


def resolve_config_registers(view: FlattenedKernelView) -> list[RegisterBindingEntry]:
    """Resolve config-role registers from ParameterResolver namespace.

    For each register with role="config", look up its name in the sub-kernel's
    resolved parameter namespace. If found, emit a RegisterBindingEntry.
    """
    bindings: list[RegisterBindingEntry] = []

    for top_iface_name in view.external_interfaces():
        for sub_name, reg, abs_offset in view.registers_for_interface(top_iface_name):
            if reg.role != "config":
                continue
            sub = view.sub_kernels[sub_name]
            if sub._resolver is None:
                continue
            ns = sub._resolver.namespace
            if reg.name not in ns:
                continue
            value = ns[reg.name]
            if not isinstance(value, (int, float)):
                continue
            bindings.append(
                RegisterBindingEntry(
                    register_name=f"{sub_name}.{reg.name}",
                    kernel_path=f"{view.name}.{sub_name}.{top_iface_name}",
                    interface_name=top_iface_name,
                    absolute_offset=abs_offset,
                    auto_bind=AutoBindSpec(param=reg.name),
                    resolved_value=int(value),
                )
            )
    return bindings


def resolve_auto_binds(view: FlattenedKernelView) -> list[RegisterBindingEntry]:
    """Resolve all auto_bind registers for the flattened view."""
    bindings: list[RegisterBindingEntry] = []

    for top_iface_name in view.external_interfaces():
        for sub_name, reg, abs_offset in view.registers_for_interface(top_iface_name):
            if not reg.auto_bind:
                continue
            value = _compute_auto_bind_value(
                reg.auto_bind, sub_name, view
            )
            bindings.append(
                RegisterBindingEntry(
                    register_name=f"{sub_name}.{reg.name}",
                    kernel_path=f"{view.name}.{sub_name}.{top_iface_name}",
                    interface_name=top_iface_name,
                    absolute_offset=abs_offset,
                    auto_bind=reg.auto_bind,
                    resolved_value=value,
                )
            )
    return bindings


def _compute_auto_bind_value(
    bind_spec: AutoBindSpec,
    sub_kernel_name: str,
    view: FlattenedKernelView,
) -> int:
    if bind_spec.value == "address":
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        addr = exposed.address
        if addr is None:
            # HW path: AXI4 MM tensors get real addresses at runtime
            # via BO allocation. Use placeholder 0; CommandInterpreter
            # substitutes with bo.address() through addr_bindings.
            addr = 0
        if bind_spec.bits:
            hi, lo = parse_bit_range(bind_spec.bits)
            return (addr >> lo) & ((1 << (hi - lo + 1)) - 1)
        return addr

    elif bind_spec.value == "size_bytes":
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        return exposed._serialized_size

    elif bind_spec.value == "size_beats":
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        iface = view.top_spec.get_interface(exposed.top_interface)
        return exposed._serialized_size // (iface.data_width // 8)

    elif bind_spec.value == "size_elements":
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        return exposed.element_count

    elif bind_spec.param:
        sub = view.sub_kernels[sub_kernel_name]
        return sub._resolver.resolve(bind_spec.param)

    elif bind_spec.expr:
        sub = view.sub_kernels[sub_kernel_name]
        return sub._resolver.resolve(bind_spec.expr)

    else:
        raise BindingError(
            f"auto_bind spec has no resolvable value: {bind_spec}"
        )
