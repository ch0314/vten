"""BFM configuration synthesis — Stage 6b of the compile pipeline.

Extracted from RuntimeEngine to keep engine.py focused on pipeline orchestration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vten.runtime.ir import BFMConfig
from vten.spec.models import DEFAULT_DATA_WIDTH, Protocol

if TYPE_CHECKING:
    from vten.runtime.flattener import FlattenedKernelView
    from vten.runtime.ir import Command


def synthesize_bfm_configs(
    view: FlattenedKernelView,
    commands: list[Command],
    buffer_ids: dict[str, int],
) -> list[BFMConfig]:
    """Build BFM configuration list from the flattened kernel view.

    One BFMConfig per external DUT interface (AXI4/AXI4S/AXI4L).
    Array interfaces are expanded into N individual BFM entries.
    """
    bfm_configs: dict[str, BFMConfig] = {}

    for top_iface_name in view.external_interfaces():
        iface_spec = view.top_spec.get_interface(top_iface_name)

        if iface_spec.protocol in (Protocol.AXI4, Protocol.AXI4S):
            address_ranges: list[tuple[int, int, int]] = []
            for exposed in view.tensors_for_interface(top_iface_name):
                if exposed.address is not None:
                    address_ranges.append(
                        (
                            exposed.address,
                            exposed._serialized_size,
                            buffer_ids[exposed.name],
                        )
                    )

            if iface_spec.array:
                for flat_name in iface_spec.array.flat_names(top_iface_name):
                    bfm_configs[flat_name] = BFMConfig(
                        interface_name=flat_name,
                        protocol=iface_spec.protocol,
                        data_width=iface_spec.data_width or DEFAULT_DATA_WIDTH,
                        role="slave" if iface_spec.protocol == Protocol.AXI4 else "master",
                        address_ranges=sorted(address_ranges),
                    )
            else:
                bfm_configs[top_iface_name] = BFMConfig(
                    interface_name=top_iface_name,
                    protocol=iface_spec.protocol,
                    data_width=iface_spec.data_width or DEFAULT_DATA_WIDTH,
                    role="slave" if iface_spec.protocol == Protocol.AXI4 else "master",
                    address_ranges=sorted(address_ranges),
                )
        elif iface_spec.protocol == Protocol.AXI4L:
            bfm_configs[top_iface_name] = BFMConfig(
                interface_name=top_iface_name,
                protocol=Protocol.AXI4L,
                data_width=32,
                role="master",
            )

    return list(bfm_configs.values())
