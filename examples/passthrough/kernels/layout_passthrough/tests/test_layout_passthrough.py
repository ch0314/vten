"""TestScenario for the layout_passthrough kernel.

Reuses the already-passing rtl/passthrough.sv DUT. The kernel adds a symmetric
layout_data_in()/unlayout_data_out() (torch.flip) hook pair, so a bit-exact
verification still succeeds. See layout_passthrough_kernel.py for the argument.

VERIFY with a real backend:
    vten run --kernel layout_passthrough --test TestLayoutPassthrough \
             --backend verilator --verify
"""

from vten.cli.scenario import TestScenario


class TestLayoutPassthrough(TestScenario):
    kernel = "layout_passthrough"

    # Sweep a couple of sizes to show the round-trip holds for any N.
    configs = [
        {"name": "default"},          # N=1024
        {"name": "small_n", "N": 32},  # one beat
    ]
