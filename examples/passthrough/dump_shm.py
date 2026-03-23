#!/usr/bin/env python3
"""Dump SHM image contents for debugging."""
import os, sys, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.dirname(__file__))

from vten.runtime.context import ExecutionContext
from vten.runtime.shm import CMD_SLOT_SIZE, BUF_DESC_SIZE, CONTROL_SIZE
from kernels.passthrough_kernel import PassthroughKernel

ctx = ExecutionContext(project_params={"N": 64})
ki = ctx.instantiate(PassthroughKernel, N=64)
ki.generate_inputs(seed=42)

h_load = ctx.load_tensor(ki.data_in)
h_push = ctx.push_tensor(ki.data_in, dep=h_load)
h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)

from vten.runtime.engine import RuntimeEngine
engine = RuntimeEngine(
    kernels=ctx._kernels, ops=ctx._pending_ops,
    project_params=ctx._project_params, alias_registry=ctx._alias_registry,
)
compiled = engine.compile()
img = compiled.shm_image

print(f"SHM image size: {len(img)} bytes")

# Control region
magic = struct.unpack_from("<I", img, 0)[0]
version = struct.unpack_from("<I", img, 4)[0]
num_cmds = struct.unpack_from("<I", img, 0x10)[0]
num_bufs = struct.unpack_from("<I", img, 0x14)[0]
cmd_off = struct.unpack_from("<Q", img, 0x18)[0]
stats_off = struct.unpack_from("<Q", img, 0x20)[0]
bufdesc_off = struct.unpack_from("<Q", img, 0x28)[0]
data_off = struct.unpack_from("<Q", img, 0x30)[0]
flags = struct.unpack_from("<I", img, 0x88)[0]
print(f"Magic=0x{magic:08X} Version=0x{version:08X} flags=0x{flags:X}")
print(f"num_cmds={num_cmds} num_bufs={num_bufs}")
print(f"cmd_off=0x{cmd_off:X} stats_off=0x{stats_off:X} bufdesc_off=0x{bufdesc_off:X} data_off=0x{data_off:X}")

# Commands — matching pack_command_slot layout
for i in range(num_cmds):
    base = cmd_off + i * CMD_SLOT_SIZE
    opcode = struct.unpack_from("<H", img, base + 0x00)[0]
    cmd_id = struct.unpack_from("<H", img, base + 0x02)[0]
    iface_id = struct.unpack_from("<H", img, base + 0x04)[0]
    proto = struct.unpack_from("<B", img, base + 0x06)[0]
    role = struct.unpack_from("<B", img, base + 0x07)[0]
    buf_id = struct.unpack_from("<H", img, base + 0x08)[0]
    probe = struct.unpack_from("<B", img, base + 0x0A)[0]
    flags_b = struct.unpack_from("<B", img, base + 0x0B)[0]
    size = struct.unpack_from("<I", img, base + 0x0C)[0]
    phys_addr = struct.unpack_from("<Q", img, base + 0x10)[0]
    reg_off = struct.unpack_from("<I", img, base + 0x18)[0]
    reg_val = struct.unpack_from("<I", img, base + 0x1C)[0]
    reg_mask = struct.unpack_from("<I", img, base + 0x20)[0]
    reg_exp = struct.unpack_from("<I", img, base + 0x24)[0]
    golden_buf = struct.unpack_from("<H", img, base + 0x28)[0]
    num_dep = struct.unpack_from("<B", img, base + 0x2A)[0]
    num_cdep = struct.unpack_from("<B", img, base + 0x2B)[0]
    deps = struct.unpack_from("<4H", img, base + 0x2C)
    cdeps = struct.unpack_from("<4H", img, base + 0x34)

    print(f"\nCmd {i}: opcode={opcode} cmd_id={cmd_id} iface_id={iface_id} "
          f"proto={proto} role={role} buf_id={buf_id} size={size}")
    print(f"  flags=0x{flags_b:02X} probe={probe} golden_buf={golden_buf}")
    print(f"  deps({num_dep}): {deps[:num_dep]} cdeps({num_cdep}): {cdeps[:num_cdep]}")
    print(f"  phys=0x{phys_addr:X} reg: off=0x{reg_off:X} val=0x{reg_val:X}")

# Buffer descriptors
for i in range(num_bufs):
    base = bufdesc_off + i * BUF_DESC_SIZE
    buf_bid = struct.unpack_from("<H", img, base + 0x00)[0]
    direction = struct.unpack_from("<B", img, base + 0x02)[0]
    buf_flags = struct.unpack_from("<B", img, base + 0x03)[0]
    buf_size = struct.unpack_from("<I", img, base + 0x04)[0]
    doff = struct.unpack_from("<Q", img, base + 0x08)[0]
    print(f"\nBuf {i}: id={buf_bid} dir={direction} size={buf_size} data_offset=0x{doff:X}")

# Also print the commands from the engine
print("\n\n--- Compiled commands ---")
for cmd in compiled.commands:
    print(f"  {cmd}")

print(f"\nBFM configs: {compiled.bfm_configs}")
