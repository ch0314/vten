# Contributing to vTen

Thanks for your interest in vTen! This project is maintained by a single
researcher at SNU, so contributions of all sizes are welcome — bug reports,
documentation fixes, new examples, and kernels alike. This guide covers the
development setup, how to run the tests, and what to expect from the review
process.

## Development setup

```bash
git clone <repo-url> vten
cd vten
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]      # dev extra = pytest + pytest-cov
```

Python >= 3.10 and PyTorch >= 2.0 are required (installed automatically as
dependencies). That is enough for the vast majority of the test suite — no
simulator needed.

### Optional simulators

- **Verilator >= 5.0** with `--timing` support — needed for the real-RTL
  hardware-functional tests (`tests/hw_functional`). Fully open source; on
  Ubuntu 24.04, `apt install verilator` is enough. (Ubuntu 22.04 ships
  Verilator 4.x, which does not properly support `--timing`.)
- **Vivado (xsim)** — optional. Tests marked `xsim` are skipped automatically
  when neither `vivado` nor `xsim` is on `PATH`, so you don't need Vivado to
  develop or to get a green run.
- **XRT / Alveo FPGA** — only for real-hardware runs; never required for tests.

## Running the tests

```bash
# CPU-only suite — fast, no simulator required. Run this before every PR.
pytest -m "not xsim" -q

# Hardware-functional suite — requires Verilator >= 5.0
pytest tests/hw_functional -q

# xsim-marked E2E tests — require Vivado; auto-skip without it
pytest -m xsim -q
```

CI (`.github/workflows/tests.yml`) runs `pytest -m "not xsim"` on Python
3.10–3.12 plus the `tests/hw_functional` suite under Verilator on
ubuntu-24.04. **CI must be green on pull requests.**

### Simulator hygiene

If you run RTL simulations (via `vten run` or while developing tests), always
keep runs bounded: the framework's `timeout_ms` watchdog (set in `vten.toml`,
see the [CLI Reference](docs/cli_reference.md)) kills a hung simulation after
the given number of milliseconds. Never set `timeout_ms = 0` — that disables
the watchdog and a hung simulator will run forever.

## Project layout

- `vten/` — the framework: frontend (`kernel/`, `dsl/`, `spec/`), runtime
  (`runtime/`), backends (`backend/`), SystemVerilog harness and codegen
  (`sv/`, `codegen/`, `templates/`), and the CLI (`cli/`).
- `tests/` — pytest suite; `tests/hw_functional/` runs real Verilator sims.
- `examples/` — runnable example projects, from "hello world" to multi-IP
  pipelines.
- `docs/` — user and developer guides.

For how the pieces fit together — the three-layer design, the compile
pipeline, and the Command[] IR — read [docs/architecture.md](docs/architecture.md).

## How to contribute

- **Open an issue first for larger changes** (new features, backend work,
  API changes). vTen has a solo maintainer, so review bandwidth is limited —
  agreeing on the direction before you write code saves everyone time, and
  responses may take a few days. Please be patient!
- **Small fixes** (typos, doc corrections, obvious bugs) can go straight to a
  pull request.
- **Include tests** for any behavior change. The existing suite under
  `tests/` shows the conventions; simulator-dependent tests belong in
  `tests/hw_functional/` or behind the `xsim` marker.
- **Keep commits focused** — one logical change per commit makes review much
  easier.
- Bug reports are valuable too: please include your Python/PyTorch versions,
  simulator (if any), and a minimal reproduction.

## Adding a kernel or example

A kernel is two files — a `kernel_spec.yaml` interface spec and a
`<name>_kernel.py` class — plus a `TestScenario`; the
[Kernel Guide](docs/kernel_guide.md) walks through all of it, and
[examples/README.md](examples/README.md) maps each framework feature to the
example that teaches it. New examples should follow the existing structure
(own `vten.toml`, `kernels/`, `rtl/`, and a README) and run on the open-source
`verilator` backend where possible so everyone can execute them.

## License

vTen is licensed under the [MIT License](LICENSE). By submitting a
contribution, you agree that it will be distributed under the same license.
