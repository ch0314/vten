# vTen

Tensor-centric verification framework for Domain-Specific Accelerators (DSA).

vTen lets you define verification scenarios in Python using tensor-level abstractions,
then compiles them into cycle-accurate RTL simulations via BFM-driven testbenches.

## Features

- **Tensor-first API** — Define inputs/outputs as PyTorch tensors with shape, dtype, and interface binding
- **8-stage compile pipeline** — DSL recording → IR lowering → SHM image generation, fully automated
- **Multi-protocol BFMs** — AXI4-Stream, AXI4 (memory-mapped), AXI4-Lite with cycle-accurate handshake
- **CompositeKernel** — Compose multi-IP pipelines with `>>` connection syntax
- **Auto-verification** — `forward()` golden reference vs. RTL output, bit-exact comparison
- **Multi-backend** — xsim (Vivado), Verilator, XRT (real FPGA), CPU emulation
- **Inference API** — `InferenceSession` for deploying kernels on FPGA hardware

## Quick Start

```bash
# Install
pip install -e .

# Create a project
vten init my_project --backend xsim

# Build (codegen + SV compile)
vten build --kernel my_accel

# Run test
vten run --kernel my_accel --test TestBasic --verify

# View results
vten report
```

## Minimal Example

**1. Define the interface** (`kernels/my_accel/kernel_spec.yaml`):

```yaml
kernel_name: my_accel
rtl_top: my_accel

interfaces:
  data_in:
    protocol: axi4_stream
    rtl_port: s_axis_data
    tensor: data_in
    packing: { element_width: 8, elements_per_beat: 32 }

  data_out:
    protocol: axi4_stream
    rtl_port: m_axis_data
    tensor: data_out
    packing: { element_width: 8, elements_per_beat: 32 }
```

**2. Write the kernel** (`kernels/my_accel/my_accel_kernel.py`):

```python
import torch
from vten import Kernel, Tensor, Direction

class MyAccelKernel(Kernel):
    spec = "kernels/my_accel/kernel_spec.yaml"

    data_in = Tensor(
        shape=("${N}",), dtype=torch.uint8,
        interface="data_in", direction=Direction.HOST_TO_DEV,
    )
    data_out = Tensor(
        shape=("${N}",), dtype=torch.uint8,
        interface="data_out", direction=Direction.DEV_TO_HOST,
    )

    def forward(self, data_in):
        return {"data_out": data_in}  # Golden reference

    def run(self, ctx):
        h = ctx.configure(self)
        ctx.push_tensor(self.data_in, dep=h)
        ctx.pull_tensor(self.data_out)
```

**3. Write the test** (`kernels/my_accel/tests/test_my_accel.py`):

```python
from vten import TestScenario

class TestBasic(TestScenario):
    kernel = "my_accel"
    configs = [{"N": 1024}]
```

## Project Structure

```
my_project/
├── vten.toml              # Project config (parameters, backend, RTL paths)
├── rtl/                    # RTL source files
├── kernels/
│   └── my_accel/
│       ├── kernel_spec.yaml   # Interface specification
│       ├── my_accel_kernel.py # Kernel class (tensors + golden + execution)
│       └── tests/
│           └── test_my_accel.py  # Test scenarios
├── build/                  # Generated artifacts
└── results/                # Test results
```

## Documentation

| Document | Description |
|----------|-------------|
| [Kernel Guide](docs/kernel_guide.md) | DUT writing, kernel_spec.yaml, Kernel class, troubleshooting |
| [CompositeKernel Guide](docs/composite_guide.md) | Multi-IP composition and connection wiring |
| [Testing Guide](docs/testing_guide.md) | TestScenario, model configs, verification workflow |
| [CLI Reference](docs/cli_reference.md) | All commands, options, and usage examples |
| [Architecture](docs/architecture.md) | System architecture and compile pipeline |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- A simulator for real-RTL runs (either one; the same tests run on both):
  - **Verilator >= 5.0** with `--timing` support — fully open source, no Vivado
    required. On Ubuntu 24.04, `apt install verilator` is enough.
  - **Vivado** (for the `xsim` backend).

## Contributing

Bug reports, fixes, and new examples are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, test
instructions, and PR guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
