"""Probe mismatch reporting — detailed error analysis and element-level diffs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vten.errors import ProbeMismatchError

logger = logging.getLogger(__name__)


def enrich_stats(
    stats: list,
    compiled: object | None,
) -> list[dict]:
    """Build enriched command stats dicts from CmdStats + CompiledResult."""
    from vten.reporting import build_command_metadata, merge_stats_with_metadata

    if compiled is not None and compiled.commands:
        metadata = build_command_metadata(compiled)
        enriched = merge_stats_with_metadata(stats, metadata)
        return [e.to_dict() for e in enriched]

    # Fallback: no CompiledResult available (pre-built SHM path)
    from vten.reporting import _status_name

    return [
        {
            "cmd_id": s.cmd_id,
            "status": s.status,
            "status_name": _status_name(s.status),
            "issue_cycle": s.issue_cycle,
            "commit_cycle": s.commit_cycle,
            "latency_cycles": s.latency_cycles,
            "active_cycles": s.active_cycles,
            "stall_cycles": s.stall_cycles,
            "total_beats": s.total_beats,
        }
        for s in stats
    ]


def report_probe_mismatch(
    pme: ProbeMismatchError,
    results_dir: Path,
    ctx,
    cfg_idx: int,
    total_cfgs: int,
) -> None:
    """Report ProbeMismatchError with dtype-aware element info."""
    # Resolve tensor name and dtype from compiled context
    tensor_name = "unknown"
    dtype_str = ""
    packing = None
    compiled = getattr(ctx, "_last_compiled", None)
    if compiled and hasattr(compiled, "buffer_ids") and hasattr(compiled, "commands"):
        # Reverse map: cmd_id -> buffer_id -> tensor_name
        cmd_bid = None
        for cmd in compiled.commands:
            if cmd.cmd_id == pme.cmd_id:
                cmd_bid = cmd.buffer_id
                break
        if cmd_bid is not None:
            bid_to_name = {bid: name for name, bid in compiled.buffer_ids.items()}
            raw_name = bid_to_name.get(cmd_bid, "unknown")
            tensor_name = raw_name.split(":")[0] if ":" in raw_name else raw_name

        # Get dtype and packing from flattened view
        view = compiled.flattened_view
        if view:
            exposed = view.exposed_tensors.get(tensor_name)
            if exposed and exposed.origin_tensor:
                dtype_str = str(exposed.origin_tensor.dtype).replace("torch.", "")
            iface_name = exposed.top_interface if exposed else None
            if iface_name:
                iface = view.top_spec.get_interface(iface_name)
                packing = iface.packing if iface else None

    # Parse mismatches.jsonl for element-level detail
    mismatch_file = results_dir / "mismatches.jsonl"
    mismatches = []
    if mismatch_file.exists():
        try:
            for line in mismatch_file.read_text().strip().splitlines():
                mismatches.append(json.loads(line))
        except Exception:
            pass

    # Build readable message
    lines = [f"probe mismatch (config {cfg_idx + 1}/{total_cfgs})"]
    lines.append(f"  tensor: {tensor_name}" + (f" ({dtype_str})" if dtype_str else ""))
    lines.append(f"  cmd_id: {pme.cmd_id}")

    if mismatches and packing:
        m = mismatches[0]
        beat = m.get("beat", 0)
        # Compute element indices from beat index
        epb = packing.elements_per_beat
        elem_start = beat * epb
        elem_end = elem_start + epb - 1
        lines.append(f"  first mismatch: beat {beat} (elements [{elem_start}..{elem_end}])")

        # Show expected vs actual bytes interpreted as dtype elements
        try:
            exp_hi = int(m.get("expected_hi", "0"), 16)
            exp_lo = int(m.get("expected_lo", "0"), 16)
            act_hi = int(m.get("actual_hi", "0"), 16)
            act_lo = int(m.get("actual_lo", "0"), 16)
            exp_bytes = exp_hi.to_bytes(4, "big") + exp_lo.to_bytes(4, "big")
            act_bytes = act_hi.to_bytes(4, "big") + act_lo.to_bytes(4, "big")

            import struct
            import torch
            ew = packing.element_width
            dtype_torch = None
            if exposed and exposed.origin_tensor:
                dtype_torch = exposed.origin_tensor.dtype

            # Show first few differing elements
            elem_size = ew // 8
            if elem_size > 0:
                n_show = min(epb, len(exp_bytes) // elem_size, 8)
                exp_vals = _unpack_elements(exp_bytes, elem_size, n_show, dtype_torch)
                act_vals = _unpack_elements(act_bytes, elem_size, n_show, dtype_torch)
                diff_indices = [
                    i for i in range(n_show)
                    if exp_vals[i] != act_vals[i]
                ]
                if diff_indices:
                    for i in diff_indices[:4]:
                        lines.append(
                            f"    [{elem_start + i}]: expected={exp_vals[i]}, "
                            f"actual={act_vals[i]}"
                        )
                    if len(diff_indices) > 4:
                        lines.append(f"    ... and {len(diff_indices) - 4} more")
        except Exception:
            # Fall back to raw hex
            lines.append(
                f"    expected: 0x{m.get('expected_hi','')}{m.get('expected_lo','')}"
            )
            lines.append(
                f"    actual:   0x{m.get('actual_hi','')}{m.get('actual_lo','')}"
            )

        if len(mismatches) > 1:
            lines.append(f"  total mismatches logged: {len(mismatches)}")
    elif mismatches:
        m = mismatches[0]
        lines.append(f"  beat {m.get('beat', '?')}, cycle {m.get('cycle', '?')}")
        lines.append(
            f"    expected: 0x{m.get('expected_hi','')}{m.get('expected_lo','')}"
        )
        lines.append(
            f"    actual:   0x{m.get('actual_hi','')}{m.get('actual_lo','')}"
        )

    logger.error("\n".join(lines))


def _unpack_elements(
    raw: bytes, elem_size: int, count: int, dtype=None,
) -> list:
    """Unpack raw bytes into element values based on dtype."""
    import struct
    import torch

    values = []
    for i in range(count):
        chunk = raw[i * elem_size : (i + 1) * elem_size]
        if len(chunk) < elem_size:
            break
        if dtype == torch.float32 and elem_size == 4:
            values.append(round(struct.unpack("<f", chunk)[0], 6))
        elif dtype == torch.float16 and elem_size == 2:
            values.append(round(struct.unpack("<e", chunk)[0], 4))
        elif dtype == torch.int32 and elem_size == 4:
            values.append(struct.unpack("<i", chunk)[0])
        elif dtype == torch.int16 and elem_size == 2:
            values.append(struct.unpack("<h", chunk)[0])
        elif elem_size == 1:
            values.append(chunk[0])
        elif elem_size == 2:
            values.append(int.from_bytes(chunk, "little"))
        elif elem_size == 4:
            values.append(int.from_bytes(chunk, "little"))
        else:
            values.append(f"0x{chunk.hex()}")
    return values
