#!/usr/bin/env python3
"""mm_loopback — Level-3 Inference API demo (InferenceSession / InferenceModule).

This is the reference example for vTen's **Level-3 Inference API**. It shows how
to drive a *verified* kernel from plain Python — passing ``torch.Tensor`` inputs
and getting a device-resident ``vten.Tensor`` output back, PyTorch-eager style —
instead of going through ``vten run`` and ``TestScenario``.

It exercises, against ``MmLoopbackKernel`` (an AXI4 memory-mapped loopback):

  1. ``session.run(...)``           — single-kernel eager execution
  2. ``session.run_pipeline(...)``  — sequential chain convenience wrapper
  3. ``InferenceModule``            — a small ``nn.Module``-style wrapper
  4. ``session.cleanup()``          — release device resources

See ``README.md`` in this directory for the surrounding example, the kernel /
spec details, and the CLI (``vten run``) equivalent.

────────────────────────────────────────────────────────────────────────────────
PREREQUISITES — please read before running
────────────────────────────────────────────────────────────────────────────────
The Inference API works against any backend registered in
``vten.backend.registry`` (``cpu``, ``verilator``, ``xsim``, ``xrt``). Which one
you can use depends on what you have installed:

  * ``cpu``       — no build, no Vivado, no FPGA. Executes the kernel's
                    behavioral ``forward()`` as if it were the DUT. This is the
                    DEFAULT below so the script runs out of the box, and every
                    Inference-API call path is still exercised end to end.

  * ``verilator`` — open source, no Vivado. Requires a prior build:
                        vten build --kernel mm_loopback --backend verilator

  * ``xsim``      — requires Vivado. Requires a prior build:
                        vten build --kernel mm_loopback --backend xsim

  * ``xrt``       — the ONLY path that touches real silicon. Requires a built
                    ``.xclbin`` and either a real Alveo FPGA (``target="hw"``) or
                    Vitis hardware emulation (``target="hw_emu"``). Build it with:
                        vten build --kernel mm_loopback --backend xrt --target hw_emu
                    The xclbin path is taken from [backend.xrt] in vten.toml
                    (kernels/mm_loopback/build/xrt/mm_loopback_hw_emu.xclbin).

On SIM backends (cpu/verilator/xsim) ``run()`` returns a host-side ``Tensor``
(``.data`` populated); on the ``xrt`` HW backend it returns a device-resident
``Tensor`` (``.on_device == True``) whose ``.cpu()`` reads it back from the
device. In both cases ``.cpu()`` yields a ``torch.Tensor``.

Run it:
    python infer.py                 # default: cpu backend (no HW needed)
    python infer.py --backend xrt   # requires a built xclbin + FPGA/hw_emu

Every InferenceSession / InferenceModule call in this file mirrors the exact
signatures in ``vten/inference.py`` — no invented APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Public Level-3 API. Both are re-exported from the top-level ``vten`` package.
from vten import InferenceModule, InferenceSession

# The real kernel class lives next to its spec. Add its directory to sys.path so
# we can import it the same way the CLI's kernel discovery does.
_KERNEL_DIR = Path(__file__).resolve().parent / "kernels" / "mm_loopback"
if str(_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_DIR))

from mm_loopback_kernel import MmLoopbackKernel  # noqa: E402

# This example's project root (the directory containing vten.toml).
PROJECT_DIR = str(Path(__file__).resolve().parent)

# Kernel facts, taken directly from mm_loopback_kernel.py + kernel_spec.yaml:
#   * input  tensor name : "data_in"   (interface mem_in,  HOST_TO_DEV, uint8)
#   * output tensor name : "data_out"  (interface mem_out, DEV_TO_HOST, uint8)
#   * shape parameter     : N  (default 1024 from vten.toml [parameters])
INPUT_NAME = "data_in"
OUTPUT_NAME = "data_out"
N = 1024


def make_input(n: int = N) -> torch.Tensor:
    """A deterministic uint8 vector of shape (n,) matching data_in's dtype/shape."""
    g = torch.Generator().manual_seed(42)
    return torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8)


def demo_single_run(session: InferenceSession) -> None:
    """1) session.run(...) — single-kernel eager execution."""
    print("\n[1] session.run(MmLoopbackKernel, inputs={'data_in': x}, verify=True)")
    x = make_input()

    # run() signature (vten/inference.py):
    #   run(kernel_class, inputs=None, *, verify=False, **params) -> dict[str, Tensor]
    # ``N`` is a kernel parameter (resolves the ${N} shape); verify=True compares
    # the DUT output against the behavioral forward() golden.
    outputs = session.run(
        MmLoopbackKernel,
        inputs={INPUT_NAME: x},
        verify=True,
        N=N,
    )

    # outputs is a dict of {tensor_name: vten.Tensor}. .cpu() -> torch.Tensor
    # (host-side on SIM backends, device readback on the xrt HW backend).
    y = outputs[OUTPUT_NAME].cpu()
    print(f"    input  : shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"    output : shape={tuple(y.shape)} dtype={y.dtype}")
    # mm_loopback is the identity, so output must equal input bit-for-bit.
    print(f"    loopback exact match: {torch.equal(x, y)}")


def demo_pipeline(session: InferenceSession) -> None:
    """2) session.run_pipeline(...) — sequential chain convenience wrapper.

    run_pipeline() calls run() once per layer and chains outputs to inputs using
    the ``chain`` map. Its DEFAULT chain is {"ofm_mem": "ifm_mem"} (NPU-style
    names), so for mm_loopback we must pass the real name mapping:
    data_out -> data_in. Two loopback layers back-to-back are still the identity.
    """
    print("\n[2] session.run_pipeline(..., chain={'data_out': 'data_in'})")
    x = make_input()

    # run_pipeline() signature (vten/inference.py):
    #   run_pipeline(kernel_class, layers, inputs, per_layer_inputs=None,
    #                chain=None, verify=False) -> dict[str, Tensor]
    result = session.run_pipeline(
        MmLoopbackKernel,
        layers=[{"N": N, "name": "loop-0"}, {"N": N, "name": "loop-1"}],
        inputs={INPUT_NAME: x},
        chain={OUTPUT_NAME: INPUT_NAME},
        verify=True,
    )

    y = result[OUTPUT_NAME].cpu()
    print(f"    2-layer loopback exact match: {torch.equal(x, y)}")


class MmLoopbackModule(InferenceModule):
    """3) InferenceModule — use a kernel like an nn.Module.

    Subclass and point kernel_cls / input_name / output_name at the real kernel.
    (The base-class defaults are the NPU-style "ifm_mem"/"ofm_mem"; mm_loopback
    uses "data_in"/"data_out", so we override them.)
    """

    kernel_cls = MmLoopbackKernel
    input_name = INPUT_NAME
    output_name = OUTPUT_NAME


def demo_module(session: InferenceSession) -> None:
    """Instantiate the InferenceModule and call it like a layer."""
    print("\n[3] InferenceModule — module(x) like an nn.Module")
    x = make_input()

    # InferenceModule.__init__ signature (vten/inference.py):
    #   __init__(session, *, weight=None, bias=None,
    #            weight_name="wgt_mem", bias_name="bias_mem", **params)
    # mm_loopback has no weights/biases; extra kwargs become kernel params.
    module = MmLoopbackModule(session, N=N)

    # forward(x, *, verify=False, **extra_inputs) -> Tensor(on_device on HW)
    y_tensor = module(x, verify=True)
    y = y_tensor.cpu()  # -> torch.Tensor
    print(f"    module output exact match: {torch.equal(x, y)}")


def build_session(backend: str, target: str) -> InferenceSession:
    """Construct an InferenceSession for the chosen backend.

    InferenceSession.__init__ signature (vten/inference.py):
        __init__(backend="xrt", base_params=None, *, kernel=None,
                 target="hw", project_dir=".", log_level=None)

    ``kernel`` is used for xclbin auto-discovery on the xrt backend and is
    harmless on SIM backends. ``target`` ("hw"/"hw_emu") is only consulted by the
    xrt backend. build_params from vten.toml are auto-injected into base_params.
    """
    print(f"Creating InferenceSession(backend={backend!r}, target={target!r})")
    return InferenceSession(
        backend,
        kernel="mm_loopback",
        target=target,
        project_dir=PROJECT_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu", "verilator", "xsim", "xrt"],
        help="Inference backend (default: cpu — no build/FPGA required). "
             "verilator/xsim need a prior 'vten build'; xrt needs an xclbin "
             "and a real FPGA or Vitis hw_emu.",
    )
    parser.add_argument(
        "--target",
        default="hw_emu",
        choices=["hw", "hw_emu"],
        help="XRT target: 'hw' (real Alveo FPGA) or 'hw_emu' (Vitis hardware "
             "emulation). Ignored by non-xrt backends. Default: hw_emu.",
    )
    args = parser.parse_args()

    session = build_session(args.backend, args.target)
    try:
        demo_single_run(session)
        demo_pipeline(session)
        demo_module(session)
    finally:
        # 4) Always release device resources (frees BOs on the xrt backend;
        #    a no-op on cpu). Signature: cleanup() -> None.
        print("\n[4] session.cleanup()")
        session.cleanup()

    print("\nDone. See examples/mm_loopback/README.md for the full walkthrough.")


if __name__ == "__main__":
    main()
