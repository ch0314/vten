"""Stage 5: Register Resolution.

Spec reference: 02_runtime_engine.md §11

v2: Unified resolve_registers() replaces resolve_config_registers,
    resolve_composite_config_registers, resolve_runtime_param_registers.
    Register name matching replaces role/alias/config_map.
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
    """Result of register resolution."""

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


def resolve_registers(view: FlattenedKernelView) -> list[RegisterBindingEntry]:
    """Unified register resolution: auto_bind first, then param-name matching.

    v2 rules:
    1. auto_bind → compute value from tensor addr/size/param/expr
    2. pulse or access="ro" → skip
    3. register name in param namespace → write param value
    4. else → skip (no matching param)
    """
    bindings: list[RegisterBindingEntry] = []

    for top_iface_name in view.external_interfaces():
        for sub_name, reg, abs_offset in view.registers_for_interface(top_iface_name):
            if reg.auto_bind:
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
            elif reg.pulse or reg.access == "ro":
                continue
            else:
                # Name matching: look up reg.name in param namespace
                sub = view.sub_kernels[sub_name]
                if sub._resolver is None:
                    continue
                ns = sub._resolver.namespace
                if reg.name in ns and isinstance(ns[reg.name], (int, float)):
                    bindings.append(
                        RegisterBindingEntry(
                            register_name=f"{sub_name}.{reg.name}",
                            kernel_path=f"{view.name}.{sub_name}.{top_iface_name}",
                            interface_name=top_iface_name,
                            absolute_offset=abs_offset,
                            auto_bind=AutoBindSpec(param=reg.name),
                            resolved_value=int(ns[reg.name]),
                        )
                    )

    return bindings


# Keep legacy names as aliases for backward compatibility during migration
resolve_auto_binds = resolve_registers
resolve_config_registers = lambda view: []
resolve_composite_config_registers = lambda view: []
resolve_runtime_param_registers = lambda view: []


def _compute_auto_bind_value(
    bind_spec: AutoBindSpec,
    sub_kernel_name: str,
    view: FlattenedKernelView,
) -> int:
    if bind_spec.value == "address":
        exposed = view.resolve_auto_bind_tensor(sub_kernel_name, bind_spec.tensor)
        addr = exposed.address
        if addr is None:
            addr = 0
        # Apply byte offset (supports parameter expressions)
        if bind_spec.offset is not None:
            if isinstance(bind_spec.offset, int):
                addr += bind_spec.offset
            else:
                sub = view.sub_kernels[sub_kernel_name]
                addr += int(sub._resolver.resolve(bind_spec.offset))
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


def _find_register_absolute_offset(
    view: FlattenedKernelView,
    sub_name: str,
    iface_name: str,
    reg_offset: int,
) -> tuple[int, str] | None:
    """Find the absolute offset and top-level interface name for a sub-kernel register."""
    for m in view.interface_mappings:
        if m.sub_kernel == sub_name and m.sub_interface == iface_name:
            return m.bank_offset + reg_offset, m.top_interface
    return None
