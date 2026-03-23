#!/usr/bin/env python3
"""Passthrough E2E: compile → SHM → xsim → golden comparison."""
import os, sys, subprocess, time, struct, ctypes, ctypes.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.dirname(__file__))

from multiprocessing.shared_memory import SharedMemory
from vten.runtime.context import ExecutionContext
from vten.runtime.shm import (
    HOST_STATUS_IDLE, HOST_STATUS_CMD_READY, BUF_DESC_SIZE,
)
from kernels.passthrough_kernel import PassthroughKernel

VTEN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# ── 1. Compile: Kernel → SHM image ──
ctx = ExecutionContext(project_params={"N": 64})
ki = ctx.instantiate(PassthroughKernel, N=64)
ki.generate_inputs(seed=42)

# Save golden reference BEFORE compile (data_in will be serialized into SHM)
golden = ki.data_in.data.clone()

h_load = ctx.load_tensor(ki.data_in)
h_push = ctx.push_tensor(ki.data_in, dep=h_load)
h_pull = ctx.pull_tensor(ki.data_out, dep=h_load)

from vten.runtime.engine import RuntimeEngine
engine = RuntimeEngine(
    kernels=ctx._kernels, ops=ctx._pending_ops,
    project_params=ctx._project_params, alias_registry=ctx._alias_registry,
)
compiled = engine.compile()
shm_image = compiled.shm_image

# Parse SHM layout for later readback
data_region_off = struct.unpack_from("<Q", shm_image, 0x30)[0]
bufdesc_off = struct.unpack_from("<Q", shm_image, 0x28)[0]
num_bufs = struct.unpack_from("<I", shm_image, 0x14)[0]

# Read buffer descriptors: find data_out buffer (buf 1)
buf_descs = []
for i in range(num_bufs):
    base = bufdesc_off + i * BUF_DESC_SIZE
    buf_size = struct.unpack_from("<I", shm_image, base + 0x04)[0]
    buf_data_off = struct.unpack_from("<Q", shm_image, base + 0x08)[0]
    buf_descs.append((buf_size, buf_data_off))

print(f"SHM image: {len(shm_image)}B, {num_bufs} buffers, data_region=0x{data_region_off:X}")
for i, (sz, off) in enumerate(buf_descs):
    print(f"  buf[{i}]: size={sz}, data_offset=0x{off:X} (abs=0x{data_region_off + off:X})")

# ── 2. Create POSIX SHM + Semaphores ──
session_id = "e2e_pt_001"
shm_name = f"vten_{session_id}"

try:
    old = SharedMemory(name=shm_name, create=False)
    old.close(); old.unlink()
except Exception:
    pass

shm = SharedMemory(name=shm_name, create=True, size=len(shm_image))
shm.buf[:len(shm_image)] = shm_image
struct.pack_into("<I", shm.buf, 0x08, HOST_STATUS_IDLE)

lib = ctypes.CDLL(ctypes.util.find_library("pthread"), use_errno=True)
lib.sem_open.restype = ctypes.c_void_p
sem_h2b = lib.sem_open(f"/vten_{session_id}_h2b".encode(),
                        ctypes.c_int(os.O_CREAT), ctypes.c_uint(0o644), ctypes.c_uint(0))
sem_b2h = lib.sem_open(f"/vten_{session_id}_b2h".encode(),
                        ctypes.c_int(os.O_CREAT), ctypes.c_uint(0o644), ctypes.c_uint(0))

class _timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

def sem_timedwait(sem, timeout_s):
    lib.sem_timedwait.argtypes = [ctypes.c_void_p, ctypes.POINTER(_timespec)]
    lib.sem_timedwait.restype = ctypes.c_int
    deadline = time.time() + timeout_s
    ts = _timespec(int(deadline), int((deadline - int(deadline)) * 1_000_000_000))
    return lib.sem_timedwait(ctypes.c_void_p(sem), ctypes.byref(ts)) == 0

# ── 3. Launch xsim ──
print(f"\nLaunching xsim (session={session_id})...")
proc = subprocess.Popen(
    ["/tools/Xilinx/Vivado/2023.2/bin/xsim", "tb_top",
     "--runall",
     "--testplusarg", f"SESSION_ID={session_id}",
     "--testplusarg", "TIMEOUT_MS=10000"],
    cwd=VTEN_ROOT,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)

# ── 4. Handshake ──
print("Waiting for backend ready...")
if not sem_timedwait(sem_b2h, 15.0):
    print("TIMEOUT: backend not ready"); proc.kill(); proc.communicate(); sys.exit(1)

print("Backend ready → CMD_READY")
struct.pack_into("<I", shm.buf, 0x08, HOST_STATUS_CMD_READY)
lib.sem_post(ctypes.c_void_p(sem_h2b))

print("Waiting for backend done...")
if not sem_timedwait(sem_b2h, 30.0):
    print("TIMEOUT: backend not done")
    proc.terminate(); time.sleep(1)
    if proc.poll() is None: proc.kill()
    out, _ = proc.communicate()
    print(out.decode(errors='replace')[-3000:])
    sys.exit(1)

backend_status = struct.unpack_from("<I", shm.buf, 0x0C)[0]
print(f"Backend done! status={backend_status} ({'DONE' if backend_status == 2 else 'ERROR'})")

# ── 5. Golden comparison: read data_out from SHM ──
out_buf_size, out_buf_data_off = buf_descs[1]  # buf 1 = data_out
abs_offset = data_region_off + out_buf_data_off
data_out_bytes = bytes(shm.buf[abs_offset : abs_offset + out_buf_size])

# Original input (golden)
golden_bytes = golden.numpy().tobytes()

print(f"\n{'='*60}")
print(f"Golden comparison: data_in vs data_out ({out_buf_size} bytes)")
print(f"{'='*60}")
print(f"  data_in  (first 32B): {golden_bytes[:32].hex()}")
print(f"  data_out (first 32B): {data_out_bytes[:32].hex()}")

if data_out_bytes == golden_bytes:
    print(f"\n  PASS: data_out == data_in (all {out_buf_size} bytes match)")
else:
    mismatches = sum(1 for a, b in zip(golden_bytes, data_out_bytes) if a != b)
    print(f"\n  FAIL: {mismatches}/{out_buf_size} bytes differ")
    for i, (a, b) in enumerate(zip(golden_bytes, data_out_bytes)):
        if a != b:
            print(f"    byte[{i}]: expected 0x{a:02X}, got 0x{b:02X}")
            if i >= 16:
                print(f"    ... ({mismatches - 16} more)")
                break

# ── 6. Shutdown ──
struct.pack_into("<I", shm.buf, 0x08, 3)  # HOST_SHUTDOWN
lib.sem_post(ctypes.c_void_p(sem_h2b))
try:
    out, _ = proc.communicate(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill(); out, _ = proc.communicate()

# ── 7. Cleanup ──
shm.close(); shm.unlink()
lib.sem_close(ctypes.c_void_p(sem_h2b))
lib.sem_close(ctypes.c_void_p(sem_b2h))
lib.sem_unlink(f"/vten_{session_id}_h2b".encode())
lib.sem_unlink(f"/vten_{session_id}_b2h".encode())
