# Testing Guide

How to write `TestScenario`s, drive multi-config runs, verify RTL output
against a golden reference, use probes, and read results.

See also: [cli_reference.md](cli_reference.md) ·
[kernel_guide.md](kernel_guide.md) ·
[composite_guide.md](composite_guide.md) ·
[architecture.md](architecture.md)

---

## Contents

- [TestScenario](#testscenario)
- [Configs & the merge order](#configs--the-merge-order)
- [The verification workflow (`--verify`)](#the-verification-workflow---verify)
- [Probes](#probes)
- [Seeds & reproducibility](#seeds--reproducibility)
- [Reading results](#reading-results)
- [Ad-hoc mode](#ad-hoc-mode)

---

## TestScenario

A `TestScenario` is a **declarative** description of what to test — a kernel
name, one or more config dicts, and optional probes. Execution is handled by
[`vten.execution.execute_batch`](../vten/execution.py); the scenario itself
holds no run logic. Class definition:
[vten/cli/scenario.py](../vten/cli/scenario.py).

```python
from vten.cli.scenario import TestScenario


class TestConv3D(TestScenario):
    kernel  = "conv3d"                       # required
    configs = [                              # optional; None ⇒ one run with base params
        {"in_ch": 64,  "out_ch": 64},
        {"in_ch": 128, "out_ch": 128},
    ]
    probes = ["scale.data_out"]              # optional; declarative probe specs
    seed   = 7                               # optional; default 42
```

| Attribute | Type | Meaning |
|-----------|------|---------|
| `kernel` | `str` | Kernel name; also used to locate `kernels/<name>/<name>_kernel.py`. |
| `configs` | `list[dict] \| None` | One dict per config to run. `None` ⇒ a single run using the base parameters. |
| `probes` | `list[str] \| None` | Declarative probe specifications (see [Probes](#probes)). |
| `seed` | `int` | Default RNG seed for `generate_inputs()` (default `42`). |

Scenarios are discovered by scanning `kernels/<kernel>/tests/test_*.py` for
`TestScenario` subclasses ([vten/cli/discovery.py](../vten/cli/discovery.py)).
`--test NAME` matches by exact class name first, then case-insensitive class
name, snake_case, or filename stem; omitting `--test` runs **all** discovered
scenarios.

The simplest possible scenario is just a kernel name — the kernel's own
`run(ctx)` method supplies the DSL protocol:

```python
class TestPassthrough(TestScenario):
    kernel = "passthrough"
```

> **How execution actually works.** For each config, `execute_batch` builds a
> fresh `ExecutionContext`, instantiates the kernel with the config kwargs,
> calls `generate_inputs(seed=...)`, registers declarative probes, then calls
> the **kernel's** `inst.run(ctx)` to record the DSL protocol, and finally
> compiles + executes. The scenario contributes `kernel`, `configs`, `probes`,
> and `seed` — the per-op DSL logic lives in the *kernel*, not the scenario.

---

## Configs & the merge order

`configs` drives **per-kernel batch** runs: one `configs` list produces a batch
where every dict is compiled and executed independently, in one backend session.
Aggregated pass/fail across the batch is reported in `summary.json`.

```python
class TestScaleAdd(TestScenario):
    kernel = "scale_add"
    configs = [
        {"name": "default"},                                       # base params only
        {"name": "identity", "scale_factor": 1, "offset_value": 0},
        {"name": "small_n", "N": 32},
        {"name": "large_n", "N": 4096},
    ]
```

### Merge order

The config passed to each execution is built in
[`_run_single_test`](../vten/cli/run.py) by layering three sources, later
sources winning on key conflicts:

```
[parameters] in vten.toml          (base)
      ⊕  --config overrides         (CLI, single-dict form)
      ⊕  scenario config entry      (one dict from `configs`)
```

Concretely:

1. `base_params = config["parameters"]` (from `vten.toml`).
2. If `--config K=V ...` was given (as a single dict), it is merged onto
   `base_params` — **CLI wins** over `vten.toml`.
3. For each entry `c` in `scenario.configs`, the run config is
   `{**base_params, **c}` — **the scenario entry wins** over base + CLI.

If `scenario.configs is None`, a single run uses `base_params` (plus any
`--config` overrides). `build_params` from the project config are also added to
each run config so the parameter resolver can reach them.

```bash
# base N=1024 from vten.toml; CLI sets N=2048 unless a scenario entry overrides it:
vten run --kernel scale_add --config N=2048
```

---

## The verification workflow (`--verify`)

Pass `--verify` (CLI) or `verify=True` (Python) to compare the DUT output
against a golden reference computed from the kernel's `forward()`.

Flow:

1. `generate_inputs(seed)` fills input tensors.
2. The kernel's `run(ctx)` records `push_tensor` / `pull_tensor` / register ops.
3. After execution, each pulled output tensor's captured data is compared
   against the golden from `forward()`.
4. Comparison uses `check_match` in [vten/verifier.py](../vten/verifier.py).

### Comparison rules

`check_match` → `compare` applies dtype-dependent tolerance:

| Data | Rule |
|------|------|
| **Floating point** | `torch.allclose(hw, golden, atol=1e-6, rtol=1e-5)` |
| **Integer** | `torch.equal(hw, golden)` — **bit-exact** |

A shape mismatch always fails.

### What a `VerificationError` reports

On mismatch, `check_match` raises
[`VerificationError`](../vten/errors.py) carrying:

- `tensor` — the tensor name that failed,
- `shape` — the effective tensor shape,
- `max_diff` — maximum element-wise absolute difference.

The message additionally reports dtype, `n_diff / total` elements that differ,
and up to 4 first-mismatch entries as
`[i,j,...]: expected=<g>, actual=<hw>`:

```
Verification failed for tensor 'data_out': shape=(1024,), dtype=int8,
max_diff=3.0, 12/1024 elements differ
  [17]: expected=42, actual=45
  [40]: expected=-3, actual=0
  ... and 10 more elements differ
```

Under `vten run`, verification failures do **not** abort the batch
(`on_error="continue"`): the failing config is recorded, per-tensor results are
written to `summary.json` (`verification_results`), and the batch status becomes
`FAIL`.

```bash
vten run --kernel conv3d --verify
```

---

## Probes

Probes capture intermediate values for beat-level comparison, in addition to
final output verification. Declarative probe specs are applied to the recorded
operations by
[vten/runtime/probe_manager.py](../vten/runtime/probe_manager.py).

There are two kinds, distinguished by whether the spec contains a `.`:

- **Output probe**: `"data_out"` marks the matching `PULL_TENSOR` op as
  `probe=True`. The BFM then compares each output beat against the golden buffer
  during simulation.
- **Composite internal probe**: `"scale.data_out"` uses the dotted
  `<sub_kernel>.<tensor>` form. It probes an internal sub-kernel tensor, and the
  expected values come from the `CompositeKernel`'s chained `forward()`.

Output probes can also be requested directly in a kernel's `run(ctx)` via
`ctx.pull_tensor(..., probe=True)`; see the
[passthrough probe scenario][passthrough-probe].

[passthrough-probe]: ../examples/passthrough/kernels/passthrough/tests/test_passthrough.py

### Beat-level golden comparison in hardware

When a pull is `probe=True`, the BFM compares each received beat against the
serialized golden buffer as the simulation runs. On mismatch the C bridge writes
a `mismatches.jsonl` file into the results directory
([vten/backend/sim/base.py](../vten/backend/sim/base.py)); the Python side parses
it and raises a
[`ProbeMismatchError`](../vten/errors.py) carrying:

- `cmd_id` — the command that detected the mismatch,
- `beat_index` — beat index of the first mismatch,
- `mismatches` — list of mismatch detail dicts (cycle, beat, expected, actual).

The first few mismatches (and the `cmd_id` / `beat_index`) are surfaced into
`summary.json` under `probe_mismatch`.

```python
class TestScaleAddProbe(TestScenario):
    kernel = "scale_add"
    probes = ["scale.data_out"]     # internal composite probe
```

---

## Seeds & reproducibility

Inputs are generated by the kernel's `generate_inputs(seed)`. In `execute_batch`
the effective seed is `cfg.get("seed", scenario.seed)` — so a per-config `seed`
key overrides the scenario default. Default is `42`.

```python
class TestConv3D(TestScenario):
    kernel  = "conv3d"
    seed    = 7
    configs = [
        {"in_ch": 64},                # uses seed=7
        {"in_ch": 64, "seed": 100},   # overrides to seed=100
    ]
```

Because seeding is done with an explicit `torch.Generator` inside
`generate_inputs`, a fixed seed makes both inputs and the `forward()` golden
fully reproducible.

---

## Reading results

`vten run` writes, per scenario, into `results/<kernel>/<test>/`:

- **`summary.json`** — status and verification roll-up.
- **`stats.json`** — per-command statistics.
- **`waveform.wdb`** — optional (with `--waveform` / `--waveform-on-fail`).
- **`mismatches.jsonl`** — optional (written by probes on mismatch).

Example `summary.json`:

```json
{
  "test_name": "TestPassthrough",
  "kernel": "passthrough",
  "status": "PASS",
  "total_cycles": 42,
  "configs_run": 1,
  "configs_passed": 1,
  "verification_count": 1,
  "verification_passed": 1,
  "verification_results": [
    { "tensor": "data_out", "passed": true, "max_diff": 0.0 }
  ]
}
```

On failure, `summary.json` additionally includes `error_message`,
`error_traceback`, and (for probe failures) a `probe_mismatch` block.

Example `stats.json` command entry:

```json
{
  "cmd_id": 0, "op": "LOAD", "protocol": "axi4_stream",
  "status_name": "COMMITTED",
  "issue_cycle": 0, "commit_cycle": 0, "latency_cycles": 0,
  "total_beats": 0, "tensor": "data_in", "size": 1024,
  "sub_kernel": "read_dma"
}
```

Format the whole `results/` tree with:

```bash
vten report                 # terminal table (grouped by sub-kernel for composites)
vten report --format json   # machine-readable
vten report --format html   # HTML table
```

See the [`vten report`](cli_reference.md#vten-report) reference for details.

---

## Ad-hoc mode

Passing `--config` as a **list of configs** (i.e. a module ref that resolves to
a list, or a sliced module ref) makes `vten run` **bypass `TestScenario`
entirely**. The kernel class is discovered directly from
`kernels/<name>/<name>_kernel.py`, all configs run via `execute_batch`, and
results land under `results/<kernel>/adhoc/`
([`_run_adhoc`](../vten/cli/run.py)).

```bash
# a module ref that resolves to list[dict] → ad-hoc batch, no TestScenario:
vten run --kernel npu --config model_configs:UNET_3D

# a slice → list → ad-hoc:
vten run --kernel npu --config model_configs:UNET_3D[0:4]

# with per-config overrides merged into every entry:
vten run --kernel npu --config model_configs:UNET_3D in_depth=4 --verify
```

A single-dict `--config` (K=V pairs, a JSON object, or an indexed module ref)
is **not** ad-hoc — it merges as a `config_overrides` layer onto the scenario
configs (see [Configs & the merge order](#configs--the-merge-order)). The
distinction is dict vs. list; see the
[`--config` grammar](cli_reference.md#the---config-grammar).
