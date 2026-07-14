"""TestScenario for the chunk_passthrough kernel.

Reuses the already-passing rtl/passthrough.sv DUT. The kernel's run() drains
data_out with chunks=4 (host-side read splitting), so a bit-exact verification
still succeeds. See chunk_passthrough_kernel.py for why chunks= needs no RTL.

The chunked pull lives in the KERNEL's run() (not this scenario), because the
CLI executes the kernel's run() — a TestScenario is pure declarative config.

VERIFY with a real backend:
    vten run --kernel chunk_passthrough --test TestChunkPassthrough \
             --backend verilator --verify
"""

from vten.cli.scenario import TestScenario


class TestChunkPassthrough(TestScenario):
    kernel = "chunk_passthrough"
    # N default 1024 → 32 beats; chunks=4 (in the kernel run) → 8 beats/chunk.
    # N=2048 → 64 beats → 16 beats/chunk. Both divide by 4 cleanly.
    configs = [
        {"name": "default"},           # N=1024
        {"name": "double", "N": 2048},
    ]
