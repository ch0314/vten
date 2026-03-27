import sys
from pathlib import Path

from vten.cli.run import TestScenario


class TestMultiPortScatter(TestScenario):
    """Split interface E2E: block_split across 2 ports.

    Verifies that _port_buffers correctly distributes tensor data
    across split ports and reassembles on readback.
    """

    kernel = "multi_port_scatter"

    def run(self, ctx, cfg):
        kernel_dir = str(Path(__file__).resolve().parent.parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)

        from multi_port_scatter_kernel import MultiPortScatterKernel

        N = cfg.get("N", 1024)
        k = ctx.instantiate(MultiPortScatterKernel, N=N)
        k.generate_inputs(seed=42)

        h_send = ctx.send_tensor(k.data_in)
        h_recv = ctx.recv_tensor(k.data_out, dep=h_send)

        ctx.verify(h_recv, k.forward())
