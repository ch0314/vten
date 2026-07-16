# CLI Reference

Complete reference for the `vten` command-line interface and the `vten.toml`
project configuration file.

The `vten` entry point is registered as a console script in
[pyproject.toml](../pyproject.toml) (`[project.scripts] vten = "vten.cli.main:main"`).
The argparse tree is defined in [vten/cli/main.py](../vten/cli/main.py).

See also: [testing_guide.md](testing_guide.md) ·
[kernel_guide.md](kernel_guide.md) ·
[composite_guide.md](composite_guide.md) ·
[architecture.md](architecture.md)

---

## Contents

- [Global flags](#global-flags)
- [`vten init`](#vten-init)
- [`vten build`](#vten-build)
- [`vten run`](#vten-run)
- [`vten list`](#vten-list)
- [`vten report`](#vten-report)
- [The `--config` grammar](#the---config-grammar)
- [`vten.toml` schema](#vtentoml-schema)
- [Typical workflow](#typical-workflow)

---

## Global flags

Global flags are parsed on the top-level parser and apply to every subcommand.
They must appear **before** the subcommand.

| Flag | Effect |
|------|--------|
| `-v`, `--verbose` | DEBUG-level logging. |
| `-q`, `--quiet` | WARNING-level logging (suppresses INFO). |
| `--log-file PATH` | Also write the debug log to `PATH`. |

```bash
vten -v run --kernel passthrough
vten --quiet build
vten --log-file build.log build --kernel conv3d
```

> Note: `build` and `run` also accept a `-v` **after** the subcommand
> (`build -v` = `--build_verbose`, `run -v` = `--sim-verbose`). On `run`, a
> subcommand-level `-v` enables *both* simulator verbosity and Python DEBUG.
> See each command below.

Exit codes: `0` success · `1` a `VTenError` (user/config error) · `2` an
internal error (re-run with `-v` for the traceback) · `130` interrupted.

The available backend list for `--backend` choices is discovered dynamically
from the backend registry ([vten/backend/registry.py](../vten/backend/registry.py)):
**`xsim`, `verilator`, `xrt`, `cpu`**.

---

## `vten init`

Create a project skeleton, or add a kernel / backend to an existing project.
Handler: [vten/cli/init_cmd.py](../vten/cli/init_cmd.py).

```
vten init <project_dir> [--kernel NAME] [--backend {xsim,verilator,xrt,cpu}] [--add-backend NAME]
```

| Argument | Description |
|----------|-------------|
| `project_dir` | Directory to create (works on new or existing dirs). |
| `--backend B` | Backend for a **new** project. Default: `xsim`. Writes that backend's config template. |
| `--kernel NAME` | Add a kernel subdirectory to an existing project instead of initializing. |
| `--add-backend NAME` | Append a `[backend.<NAME>]` section (and its dirs) to an existing project's `vten.toml`. |

Behavior:

- **Full init** (no `--kernel`/`--add-backend`): creates common dirs
  `rtl/`, `kernels/`, `results/` plus backend-specific dirs
  (e.g. `xsim` → `build/vivado_proj`, `build/lib`, `ip`), and writes
  `vten.toml` only if it does not already exist.
- **`--kernel NAME`**: creates `kernels/<NAME>/` with `kernel_spec.yaml`,
  `<NAME>_kernel.py`, `tests/test_<NAME>.py`, and per-kernel
  `build/generated`, `build/shm` skeletons. Existing files are not overwritten.
- **`--add-backend NAME`**: errors if `vten.toml` is missing or the section
  already exists; otherwise appends the backend template.

```bash
vten init my_project --backend xsim
vten init my_project --kernel conv3d       # add a kernel
vten init my_project --add-backend verilator
```

---

## `vten build`

Codegen + compile for a project or a single kernel. `build`
([vten/cli/build.py](../vten/cli/build.py)) resolves the selected backend and
then runs that backend's `BuildPipeline`.

```
vten build [--project DIR] [--kernel NAME] [--backend B]
           [--stage S] [--upto S] [--target {hw,hw_emu}]
           [--force] [--clean] [--skip-compile] [-v] [--config K=V ...]
```

| Flag | Description |
|------|-------------|
| `--project DIR` | Project directory (contains `vten.toml`). Default: `.`. |
| `--kernel NAME` | Build only this kernel (default: build all). |
| `--backend B` | Override backend. Priority: `--backend` > `[project].default_backend` > `xsim`. |
| `--stage S` | Run exactly one stage. |
| `--upto S` | Run all stages up to and including `S`. |
| `--target {hw,hw_emu}` | XRT build target; overrides `[backend.xrt].target` in `vten.toml`. |
| `--force` | Ignore the build cache — full rebuild. |
| `--clean` | Remove build artifacts before building. |
| `--skip-compile` | Run codegen only (skip the compile stage). |
| `-v` | DEBUG output (`--build_verbose`). |
| `--config K=V ...` | Config overrides passed to the build (see grammar below). |

### Build stages

Each backend's `BuildPipeline` defines its own ordered stage list.

xsim pipeline ([vten/build/xsim_build.py](../vten/build/xsim_build.py)):

```
project_setup → dpi_c → codegen → compile_order → compile
```

| Stage | Granularity | What it does |
|-------|-------------|--------------|
| `project_setup` | project | Vivado project creation (cached). |
| `dpi_c` | project | Build the DPI-C shared library via gcc (cached). |
| `codegen` | per-kernel | Jinja2 → generated SystemVerilog testbench (cached). |
| `compile_order` | per-kernel | Vivado `get_compile_order` (cached). |
| `compile` | per-kernel | `xvlog`/`xelab` compile of the testbench. |

verilator pipeline ([vten/build/verilator_build.py](../vten/build/verilator_build.py)):

```
dpi_c → codegen → verilate → make
```

| Stage | Granularity | What it does |
|-------|-------------|--------------|
| `dpi_c` | project | Build the SHM-bridge shared library via gcc (cached; no Vivado includes). |
| `codegen` | per-kernel | Jinja2 → generated SystemVerilog testbench (same as xsim). |
| `verilate` | per-kernel | `verilator --cc --exe --main --timing` → C++ model in `build/obj_dir/`. |
| `make` | per-kernel | `make -C obj_dir` → the standalone `Vtb_top` simulator binary. |

Expect a fresh verilator build to take on the order of **~5 minutes per
kernel**, dominated by the g++ compile of the verilated model in the `make`
stage; unchanged stages are cached on rebuild. Note that the verilator
`codegen` stage requires a `kernels/<name>/kernel_spec.yaml`, so **composite
kernels (which have no spec of their own) currently build only under the xsim
pipeline** — unit kernels only under verilator.

`--stage` / `--upto` argparse choices are the xsim stage names above
([vten/cli/main.py](../vten/cli/main.py)).

`--skip-compile` runs codegen but not `compile`.

### build `--config` form

For `build`, `--config` is a **simple `K=V`** form with light coercion:
purely-numeric values are cast to `int`, everything else stays a string
(see `_dispatch` in [vten/cli/main.py](../vten/cli/main.py)).

```bash
vten build --config in_ch=64 out_ch=32     # in_ch=64 (int), out_ch=32 (int)
vten build --config layout=nchw            # layout="nchw" (string)
```

`--target` is threaded through the same mechanism internally (as
`_xrt_target`) and rewrites `[backend.xrt].target`.

```bash
vten build --kernel conv3d --upto codegen           # stop after codegen
vten build --kernel conv3d --stage compile --force  # force-recompile only
vten build --backend xrt --target hw_emu            # XRT hw_emu build
```

---

## `vten run`

Discover, execute, and record results for test scenario(s).
Handler: [vten/cli/run.py](../vten/cli/run.py).

```
vten run --kernel NAME [--test SCENARIO] [--project DIR] [--backend B]
         [--waveform] [--waveform-on-fail] [--gui]
         [-v | --sim-verbose] [--verify] [--config SPEC ...]
```

| Flag | Description |
|------|-------------|
| `--kernel NAME` | **Required.** Kernel to run. |
| `--test SCENARIO` | Run one `TestScenario` by name. Omit to run **all** scenarios in `kernels/<NAME>/tests/`. |
| `--project DIR` | Project directory. Default: `.`. |
| `--backend B` | Override backend (same priority as build). |
| `--waveform` | Always dump a waveform (saved to `results/.../waveform.wdb`). |
| `--waveform-on-fail` | Dump a waveform, but delete it again if the test passes. |
| `--gui` | xsim GUI mode. |
| `-v`, `--sim-verbose` | Enable simulator verbose output and raise Python logging to DEBUG. |
| `--verify` | Auto-verify RTL output against the `forward()` golden. |
| `--config SPEC ...` | Config overrides / ad-hoc configs (see grammar below). |

Behavior notes:

- Results are written to `results/<kernel>/<test>/` as `summary.json` and
  `stats.json` (see [testing_guide.md](testing_guide.md)).
- Without `--test`, all discovered `TestScenario` subclasses are run
  sequentially in one backend session.
- A **list** of configs from `--config` switches `run` into **ad-hoc mode**,
  bypassing `TestScenario` discovery entirely (`test_name = "adhoc"`).

```bash
vten run --kernel passthrough                          # run all scenarios
vten run --kernel passthrough --test TestPassthrough --verify
vten run --kernel conv3d --backend verilator --waveform-on-fail
vten run --kernel conv3d --config in_ch=64 out_ch=32   # override scenario configs
```

---

## `vten list`

List test scenarios or kernel parameters. Handler:
[vten/cli/list_cmd.py](../vten/cli/list_cmd.py).

```
vten list tests  --kernel NAME [--project DIR]
vten list params --kernel NAME [--project DIR]
```

- **`vten list tests --kernel NAME`** — discovers `TestScenario` subclasses
  in `kernels/<NAME>/tests/` and prints each scenario name, its config count,
  and the first line of its docstring.
- **`vten list params --kernel NAME`** — parses `kernel_spec.yaml` and prints
  spec parameters (`${PARAM}` placeholders with defaults / `required`),
  build parameters, interfaces (with protocol + bound tensor), and per-interface
  registers (name, offset, fields).

Backward-compat shorthand: `vten list --kernel NAME` (no subcommand) is an
alias for `vten list tests --kernel NAME`.

```bash
vten list tests --kernel scale_add
vten list params --kernel conv3d
```

---

## `vten report`

Format the results under `results/`. Handler:
[vten/cli/report.py](../vten/cli/report.py).

```
vten report [--project-dir DIR] [--format {terminal,html,json}]
```

| Flag | Description |
|------|-------------|
| `--project-dir DIR` | Project directory containing `results/`. Default: `.`. |
| `--format` | `terminal` (default), `html`, or `json`. |

The scanner supports both the nested `results/<kernel>/<test>/summary.json`
layout produced by `run`, and a flat `results/<test>/summary.json` layout.
The `terminal` format prints a per-test status header, a per-command table
(grouped by sub-kernel for CompositeKernels), and a verification summary.

```bash
vten report
vten report --format json > report.json
vten report --format html > report.html
```

---

## The `--config` grammar

`run` and `build` both accept `--config`, but with **different** grammars.

### `run --config` (rich grammar)

Parsed by `resolve_config` in
[vten/cli/config_resolver.py](../vten/cli/config_resolver.py). Four forms:

| Form | Example | Result |
|------|---------|--------|
| **K=V pairs** | `in_ch=128 out_ch=64` | `{"in_ch": 128, "out_ch": 64}` |
| **JSON object** | `'{"in_ch": 128, "out_ch": 64}'` | parsed dict (single arg starting with `{`) |
| **Module ref** | `model_configs:UNET_MINI` | the module-level variable as-is (dict or `list[dict]`) |
| **Module ref, indexed** | `model_configs:UNET_MINI[0]` | single element (dict) |
| **Module ref, sliced** | `model_configs:UNET_MINI[1:4]` | slice (`list[dict]`) |

**Type coercion** for K=V values: integers (incl. negative) → `int`,
otherwise `float` if parseable, `true`/`false` → `bool`, else the raw string.

**Mixed form** — a module ref plus trailing K=V overrides. The module ref is
resolved first, then each K=V is merged into *every* resulting config:

```bash
vten run --kernel npu --config model_configs:UNET_3D[0] in_depth=4 in_height=4
```

The module is imported from `kernels/` (added to `sys.path`), so
`model_configs.py` living alongside your kernels is importable. A `module:VAR`
token is recognized as a module ref only when it contains `:` and no `=`.

**Single dict vs. list** matters for run semantics. A single dict is treated as
`config_overrides` and is merged onto the discovered scenario configs. A **list**
of config dicts switches `run` into ad-hoc mode, which bypasses `TestScenario`
discovery and runs those configs directly (see [testing_guide.md](testing_guide.md)).

### `build --config` (simple grammar)

Only `K=V` pairs; numeric-looking values are coerced to `int`, everything else
stays a string. No JSON, no module refs. See the [build](#vten-build) section.

---

## `vten.toml` schema

Loaded by `load_project_config` in [vten/cli/config.py](../vten/cli/config.py).
Only `[project]` is strictly required. Paths are resolved relative to the
project root (where `vten.toml` lives). Templates for each backend section live
in [vten/cli/init_cmd.py](../vten/cli/init_cmd.py).

### `[project]` (required)

```toml
[project]
name = "my_project"
version = "0.1.0"
default_backend = "xsim"   # optional; run/build fall back to "xsim" if absent
```

| Key | Meaning |
|-----|---------|
| `name` | Project name. |
| `version` | Project version. |
| `default_backend` | Backend used when `--backend` is not given. |

### `[tools]`

Unified tool-path section with per-backend fallback. `resolve_tool_path`
looks up `[backend.<B>].<tool>` first, then `[tools].<tool>`, then the
short form (`vivado_path` → `vivado`).

```toml
[tools]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
```

### `[parameters]`

Project-wide default parameters. These form the base config that
`--config` overrides and scenario `configs` merge on top of
(see [testing_guide.md](testing_guide.md)).

```toml
[parameters]
N = 1024
```

### `[backend.xsim]`

```toml
[backend.xsim]
part = "xcu250-figd2104-2L-e"
vivado_path = "/tools/Xilinx/Vivado/2023.2"
compile_options = ["-timescale", "1ns/1ps"]
glbl = false                 # link Vivado glbl for gate-level/UNISIM
xelab_libs = []              # extra -L libraries for xelab
timeout_ms = 10000           # sim TIMEOUT_MS (+testplusarg); 0 = no timeout
submit_timeout_s = 300
```

| Key | Meaning |
|-----|---------|
| `part` | Xilinx device part for the Vivado project. |
| `vivado_path` | Per-backend Vivado path (falls back to `[tools]`). |
| `compile_options` | Extra `xvlog`/`xelab` options. |
| `glbl` | Link Vivado `glbl` during elaboration. |
| `xelab_libs` | Extra `-L <lib>` entries for `xelab`. |
| `timeout_ms` | Simulator watchdog in ms (`0` disables). |
| `submit_timeout_s` | Python-side submit timeout. |

### `[backend.verilator]`

Requires Verilator **>= 5.0** (the build uses `--timing`, which 4.x does not
properly support). No Vivado install is needed.

```toml
[backend.verilator]
verilator_path = ""     # empty = use PATH
threads = 4
trace = false           # enable VCD/FST tracing
opt_level = 3
timeout_ms = 10000           # sim TIMEOUT_MS (+plusarg); 0 = no timeout
submit_timeout_s = 300
# extra_args = [...]         # extra verilator flags (appended after defaults)
# sim_models = "sim_models"  # project dir of IP behavioral models
```

| Key | Meaning |
|-----|---------|
| `verilator_path` | Verilator binary; empty = `verilator` from `PATH`. |
| `threads` | Verilation parallelism (`-j`). |
| `trace` | Waveform tracing (`--trace` at build, `+trace` at run). |
| `opt_level` | Verilator `-O<n>` optimization level. |
| `timeout_ms` | Simulator watchdog in ms (`0` disables — not recommended, a hung sim then runs forever). |
| `submit_timeout_s` | Python-side submit timeout. |
| `extra_args` | Extra `verilator` flags. |
| `sim_models` | Directory of `.v`/`.sv` behavioral models for Vivado IP (default `sim_models/`). Project models override the framework models in `vten/sv/verilator/` when module names collide. |

The pipeline always passes `--unroll-count 256 --unroll-stmts 200000` and
`-Wno-SIDEEFFECT` (plus other warning suppressions): the generated command
scheduler/controller loops over the static `MAX_CMDS` bound and must be fully
unrolled (see the comment in
[vten/build/verilator_build.py](../vten/build/verilator_build.py)).
`extra_args` is appended **after** these defaults, and for Verilator the last
occurrence of a flag wins — so a project can override them, e.g.
`extra_args = ["--unroll-count", "512"]` when raising `max_cmds`.

### `[backend.xrt]`

```toml
[backend.xrt]
platform = "/opt/xilinx/platforms/.../*.xpfm"
target = "hw_emu"            # or "hw"; overridable via build --target
clock_freq_hz = 300000000   # optional freqHz constraint per kernel
xclbin_path = "build/kernel.xclbin"
device_index = 0
kernel_name = "mm_loopback" # default IP kernel name inside the xclbin
poll_timeout_ms = 300000    # POLL_REG timeout
part = "xcu280-fsvh2892-2L-e"
```

| Key | Meaning |
|-----|---------|
| `platform` | `.xpfm` platform for `v++` link (required for XRT build). |
| `target` | `hw` or `hw_emu`. `build --target` overrides this. |
| `clock_freq_hz` | Optional per-kernel `freqHz` link constraint. |
| `xclbin_path` | Path to the `.xclbin` to load. |
| `device_index` | FPGA device index (default `0`). |
| `kernel_name` | Default IP/kernel name within the xclbin. |
| `poll_timeout_ms` | `POLL_REG` timeout (ms). Default differs for `hw` vs `hw_emu`. |
| `part` | Device part. |

### `[backend.scheduler]`

Overrides the auto-computed command-scheduler sizing. The codegen takes
`max(auto, configured)` for each ([vten/codegen/sv_generator.py](../vten/codegen/sv_generator.py)).

```toml
[backend.scheduler]
max_bfms = 8       # max BFM instances
max_ifaces = 16    # max interface slots
max_cmds = 256     # max in-flight commands
```

### `[rtl]`

```toml
[rtl]
sources = ["rtl/**/*.sv", "rtl/**/*.v"]   # glob patterns (project-relative)
include_dirs = ["rtl/include"]
top_module = "..."     # optional explicit top
tb_module = "tb_top"   # optional explicit testbench module
```

### `[[ip]]`

Zero or more IP sources (array-of-tables):

```toml
[[ip]]
source = "ip/**/*.xci"
```

### `[test]`

```toml
[test]
default_seed = 42
waveform = false
waveform_on_fail = true
```

| Key | Meaning |
|-----|---------|
| `default_seed` | Default RNG seed for input generation. |
| `waveform` | Default waveform dumping. |
| `waveform_on_fail` | Keep waveform only when a test fails. |

### Full example

From [examples/mm_loopback/vten.toml](../examples/mm_loopback/vten.toml):

```toml
[project]
name = "mm_loopback"
version = "0.1.0"
default_backend = "xsim"

[parameters]
N = 1024

[backend.xsim]
vivado_path = "/tools/Xilinx/Vivado/2023.2"
part = "xcu250-figd2104-2L-e"
compile_options = ["-timescale", "1ns/1ps"]
timeout_ms = 30000
submit_timeout_s = 300

[backend.verilator]
threads = 4
trace = false

[backend.xrt]
xclbin_path = "kernels/mm_loopback/build/xrt/mm_loopback_hw_emu.xclbin"
device_index = 0
kernel_name = "mm_loopback"
poll_timeout_ms = 300000
platform = "/opt/xilinx/platforms/.../xilinx_u280_gen3x16_xdma_1_202211_1.xpfm"
part = "xcu280-fsvh2892-2L-e"
target = "hw_emu"

[rtl]
sources = ["rtl/*.sv"]

[test]
default_seed = 42
waveform = false
waveform_on_fail = true
```

---

## Typical workflow

```bash
# 1. Create the project + a kernel
vten init my_project --backend xsim
cd my_project
vten init . --kernel conv3d
# ... edit kernels/conv3d/kernel_spec.yaml, conv3d_kernel.py, tests/ ...

# 2. Build (codegen + compile)
vten build --kernel conv3d

# 3. Run with golden verification
vten run --kernel conv3d --verify

# 4. Inspect results
vten report
```

Iterate on stages during bring-up:

```bash
vten build --kernel conv3d --upto codegen          # inspect generated SV
vten run --kernel conv3d --test TestSmall -v --waveform-on-fail
vten list tests --kernel conv3d                    # what scenarios exist?
vten list params --kernel conv3d                   # what params/interfaces?
```

> `vten spec --detect` is **not** a real command — it was described in an
> earlier design note but never implemented; there is no `spec` subcommand. See
> [paper_vs_code.md](paper_vs_code.md).
