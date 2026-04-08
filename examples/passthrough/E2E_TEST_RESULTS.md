# E2E Test Results — passthrough project

## Test Environment

- **Backend**: xsim (Vivado 2023.2)
- **Default parameters**: N=1024
- **Project directory**: `examples/passthrough/`

---

## 1. passthrough

**Purpose**: Simplest DUT. AXI4-Stream in → AXI4-Stream out (data unchanged).

**RTL**: `rtl/passthrough.sv`
- Pure combinational: `m_axis_tdata = s_axis_tdata`, handshake/tlast pass-through

**Kernel**: `PassthroughKernel`
- Tensors: `data_in` (input_stream), `data_out` (output_stream)
- `forward()` returns `data_in.data.clone()`

**Features validated**:
- AXI4-Stream PUSH/PULL
- Basic SHM → DPI-C → BFM → DUT pipeline
- Host-side verification (`ctx.run(verify=True)`)

### TestPassthrough
| Field | Value |
|-------|-------|
| Status | PASS |
| Verification | 1/1 passed |

### TestPassthroughProbe
| Field | Value |
|-------|-------|
| Status | PASS |
| Verification | 1/1 passed |
| Probe mismatch | None (output == input == golden_buf) |

**Features validated**: `probe=True` on PULL, golden_buf comparison (no mismatch expected)

---

## 2. vector_alu

**Purpose**: AXI4 read path, multi-tensor single-port, register-based operation selection.

**RTL**: `rtl/vector_alu_core.sv`
- AXI4 master reads operand_a, operand_b from DDR, computes element-wise op, writes result back
- FSM: IDLE → READ_A → READ_B → COMPUTE → WRITE → DONE
- Operations: ADD (saturating), SUB (saturating), MUL (low 8-bit)
- AXI4-Lite slave for control registers (auto-generated wrapper)

**Kernel**: `VectorAluKernel`
- Tensors: `operand_a`, `operand_b` (HOST_TO_DEV), `result` (DEV_TO_HOST) — all on single `mem_port` (AXI4)
- Address-multiplexed: 3 tensors share 1 AXI4 port, BFM routes by address (`find_entry()`)
- `forward(op_mode)` computes golden with saturating arithmetic

**Kernel spec**: `kernel_spec.yaml`
- 10 registers with 6 `auto_bind` entries for 3 tensor addresses (lo/hi split)
- `op_mode` register for operation selection
- `ctrl` pulse register (start), `status` read-only register (done)
- `memory_regions: ddr` (base=0x10000000)

**Features validated**:
- AXI4 read path (AR/R channel) — first time tested
- AXI4 write path (AW/W/B channel)
- Multi-tensor single-port AXI4 address multiplexing
- `auto_bind` address resolution (lo/hi 32-bit split)
- `write_register` with field resolution (`op_mode`, `start`)
- `poll_register` for completion
- `generate_controller: true` (AXI-Lite wrapper auto-generation)
- Command ordering: BFMs activated BEFORE DUT start trigger

### TestVectorAluAdd (op_mode=0)
| Field | Value |
|-------|-------|
| Status | PASS |
| Total cycles | ~520 |
| Verification | 1/1 passed |

### TestVectorAluSub (op_mode=1)
| Field | Value |
|-------|-------|
| Status | PASS |
| Verification | 1/1 passed |

### TestVectorAluMul (op_mode=2)
| Field | Value |
|-------|-------|
| Status | PASS |
| Verification | 1/1 passed |

### TestVectorAluProbe
| Field | Value |
|-------|-------|
| Status | PASS |
| Total cycles | 522 |
| Verification | 1/1 passed (host-side uses correct golden) |
| Probe mismatch | 30 beats — all mismatch (result=A+B vs golden_buf=operand_a) |

**Features validated**: Probe mismatch detection on non-passthrough kernel. BFM correctly reports beat-by-beat mismatches when golden_buf differs from actual output.

**Key bug found & fixed during development**:
- DECERR / xsim timeout: BFM PUSH/PULL must be dispatched BEFORE DUT start. If PUSH depends on start, BFM has no active entry when DUT issues AR → DECERR.

---

## 3. stream_scatter

**Purpose**: Mixed protocol DUT — AXI4-Stream input, dual AXI4 HBM output, AXI4-Lite control.

**RTL**: `rtl/stream_scatter_core.sv`
- AXI4-Stream input → scale x2 (saturating) → alternating write to dual AXI4 HBM ports
- Even beats → hbm_0, odd beats → hbm_1
- Combinational write-port mux based on `use_port1` toggle
- `reg_beat_count` output register (32-bit counter)

**Kernel**: `StreamScatterKernel`
- Tensors: `data_in` (input_stream, HOST_TO_DEV), `result_0` (hbm_0, DEV_TO_HOST), `result_1` (hbm_1, DEV_TO_HOST)
- Parametric expression: `"${N}//2"` for output tensor shapes
- `forward()` returns `(even_beats, odd_beats)` tuple after scatter

**Kernel spec**: `kernel_spec.yaml`
- 4 interfaces: ctrl (AXI4-Lite), input_stream (AXI4-Stream), hbm_0 (AXI4), hbm_1 (AXI4)
- `auto_bind` for dst0/dst1 addresses
- `length` register: NO auto_bind (AXI4-Stream has data_width=None, breaks size_beats calc)
- `beat_count` register: read-only, `fields: { count: "31:0" }`

**Features validated**:
- Mixed protocol: AXI4-Stream + AXI4 + AXI4-Lite in single DUT
- Multi-port AXI4 (HBM-style dual write ports)
- `ctx.barrier()` — BARRIER command
- `ctx.read_register()` — READ_REG command
- `ctx.write_register()` with manual value (length)
- `ctx.poll_register()` with dual PULL commit dependency
- Parametric expressions in tensor shapes (`"${N}//2"`)
- Dual output verification (`run(verify=True)` with 2 output tensors)

### TestStreamScatter
| Field | Value |
|-------|-------|
| Status | PASS |
| Total cycles | ~200 |
| Verification | 2/2 passed |

**Key issue found during development**:
- `auto_bind: size_beats` fails for AXI4-Stream tensors (data_width=None). Workaround: write length register manually in test.

---

## 4. broken_passthrough

**Purpose**: Deliberately buggy DUT to verify probe mismatch detection catches RTL errors.

**RTL**: `rtl/broken_passthrough_core.sv`
- Same as passthrough but XORs every byte with 0x01 (flips bit 0)
- Pure combinational corruption via generate loop

**Kernel**: `BrokenPassthroughKernel`
- Same structure as PassthroughKernel
- `forward()` returns CORRECT output (input unchanged) — mismatches expected

**Features validated**:
- Probe mismatch detection on buggy RTL
- Host-side verify failure on data corruption
- Non-fatal probe: simulation completes despite mismatches

### TestBrokenPassthrough (no probe)
| Field | Value |
|-------|-------|
| Status | FAIL |
| Verification | 0/1 passed |
| Note | Host-side verify detects corruption |

### TestBrokenPassthroughProbe (probe=True)
| Field | Value |
|-------|-------|
| Status | FAIL |
| Verification | 0/1 passed |
| Probe mismatch | All 32 beats — every byte differs by XOR 0x01 |

**xsim.log example**:
```
[PROBE MISMATCH] cycle=11 beat=0 expected=0x0217E7E3_8EDC33E6 actual=0x0316E6E2_8FDD32E7
[PROBE MISMATCH] cycle=12 beat=1 expected=0x3D2E3D3D_7C95B94B actual=0x3C2F3C3C_7D94B84A
```

**Key fix during development**:
- `vten.toml` global `top_module = "passthrough"` removed — was overriding per-kernel module name derivation in multi-kernel project.

---

## Feature Coverage Matrix

| Feature | passthrough | vector_alu | stream_scatter | broken_passthrough |
|---------|:-----------:|:----------:|:--------------:|:------------------:|
| AXI4-Stream PUSH | v | | v | v |
| AXI4-Stream PULL | v | | | v |
| AXI4 READ (AR/R) | | v | | |
| AXI4 WRITE (AW/W/B) | | v | v | |
| AXI4-Lite control | | v | v | |
| Multi-tensor single-port | | v | | |
| Multi-port AXI4 | | | v | |
| Mixed protocols | | | v | |
| auto_bind (address) | | v | v | |
| write_register | | v | v | |
| read_register | | | v | |
| poll_register | | v | v | |
| barrier | | | v | |
| probe (match) | v | | | |
| probe (mismatch) | | v | | v |
| generate_controller | | v | v | |
| Parametric expressions | | | v | |
| Host-side verify PASS (run(verify=True)) | v | v | v | |
| Host-side verify FAIL | | | | v |

## Not Yet Tested

- CompositeKernel (multi-kernel composition)
- Multi-batch execution
- probe with explicit golden_buf_id (CompositeKernel Internal)
- Waveform dump (`--waveform`)
