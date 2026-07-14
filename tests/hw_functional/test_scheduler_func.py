"""Functional tests for vten_command_scheduler — dependency-aware dispatch.

Tests verify multi-BFM concurrent dispatch, sync chain ordering,
barrier fence semantics, and error paths. The C++ driver simulates
both the controller feed side and BFM done signals.

Scheduler parameters for testing: MAX_BFMS=4, MAX_CMDS=32, MAX_IFACES=8.
Default iface_to_bfm mapping: iface 0→BFM 0, iface 1→BFM 1, etc.
"""

from __future__ import annotations

from tests.hw_functional.conftest import (
    OP_BARRIER,
    OP_LOAD,
    OP_PUSH,
    OP_PULL,
    OP_STORE,
    OP_WRITE_REG,
    PROTO_AXI4S,
    PROTO_AXI4L,
    build_shm_image,
    requires_verilator,
)

pytestmark = requires_verilator


# ── Helpers ──

def _setup(sim, commands, **kw):
    """Load SHM image, create DUT, reset, feed commands, signal feed_done."""
    num_cmds = len(commands)
    image = build_shm_image(num_commands=num_cmds, commands=commands, **kw)
    sim.load_shm_image(image)
    sim.create()
    sim.reset(5)

    for c in commands:
        sim._send({
            "cmd": "feed_cmd",
            "opcode": c["opcode"], "cmd_id": c["cmd_id"],
            "iface_id": c.get("interface_id", 0),
            "protocol": c.get("protocol", PROTO_AXI4S),
            "role": c.get("role", 0),
            "sync": 1 if c.get("sync") else 0,
            "size": c.get("size", 256),
        })
    sim._send({"cmd": "feed_done"})


def _tick(sim, n=1):
    return sim._send({"cmd": "tick", "n": n})


def _collect_dispatches(sim, ticks=5):
    """Tick multiple times and collect which BFMs received dispatches."""
    seen = {"bfm0": 0, "bfm1": 0, "bfm2": 0, "bfm3": 0}
    for _ in range(ticks):
        _tick(sim, 1)
        d = sim._send({"cmd": "get_dispatched"})
        for k in seen:
            if d[k]:
                seen[k] = 1
    return seen


def _get_dispatched(sim):
    return sim._send({"cmd": "get_dispatched"})


def _set_bfm_done(sim, bfm, cmd_id, error=0, error_code=0):
    return sim._send({
        "cmd": "set_bfm_done", "bfm": bfm,
        "done_cmd_id": cmd_id, "done_error": error, "error_code": error_code,
    })


def _get_state(sim):
    return sim._send({"cmd": "get_state"})



# ═══════════════════════════════════════════════════════════════════════════
# Multi-BFM Concurrent Dispatch
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiBFMDispatch:
    """Test cross-BFM parallelism — multiple commands dispatched simultaneously."""

    def test_two_independent_cmds_dispatch(self, scheduler_sim):
        """Two independent cmds on different BFMs both get dispatched."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1, "BFM 0 should have cmd dispatched"
        assert seen["bfm1"] == 1, "BFM 1 should have cmd dispatched"

    def test_three_bfms_concurrent(self, scheduler_sim):
        """Three cmds on three BFMs dispatch concurrently."""
        commands = [
            {"cmd_id": i, "opcode": OP_PUSH, "interface_id": i,
             "protocol": PROTO_AXI4S, "size": 256}
            for i in range(3)
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1
        assert seen["bfm1"] == 1
        assert seen["bfm2"] == 1
        assert seen["bfm3"] == 0

    def test_per_bfm_rate_limiting(self, scheduler_sim):
        """Two cmds on same BFM: second dispatches on the next cycle (one per cycle)."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)

        # Collect dispatches tick-by-tick; cmd_valid is a one-cycle pulse
        dispatch_count = 0
        for _ in range(5):
            _tick(scheduler_sim, 1)
            d = _get_dispatched(scheduler_sim)
            if d["bfm0"]:
                dispatch_count += 1

        # Both independent cmds should dispatch (bfm_used_this_cycle limits to 1/cycle)
        assert dispatch_count == 2, f"Expected 2 dispatches on BFM 0, got {dispatch_count}"

    def test_four_bfms_all_dispatched(self, scheduler_sim):
        """Four cmds on four BFMs all get dispatched."""
        commands = [
            {"cmd_id": i, "opcode": OP_PUSH, "interface_id": i,
             "protocol": PROTO_AXI4S, "size": 256}
            for i in range(4)
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1
        assert seen["bfm1"] == 1
        assert seen["bfm2"] == 1
        assert seen["bfm3"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Sync Chain Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSyncChain:
    """Test sync flag ordering: cmd with sync=1 blocks subsequent cmds."""

    def test_sync_blocks_next_cmd(self, scheduler_sim):
        """Cmd 1 should not dispatch until sync cmd 0 is committed."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256, "sync": True},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)

        # After feed_done: cmd 0 dispatches, cmd 1 blocked by sync
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1, "Cmd 0 should dispatch"
        assert seen["bfm1"] == 0, "Cmd 1 blocked by sync on cmd 0"

        # Complete cmd 0 on BFM 0
        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0)
        seen2 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen2["bfm1"] == 1, "Cmd 1 should dispatch after sync cmd 0 committed"

    def test_sync_does_not_block_itself(self, scheduler_sim):
        """A cmd with sync=1 dispatches normally — it only blocks later cmds."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256, "sync": True},
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1, "Sync cmd should dispatch normally"

    def test_double_sync_chain(self, scheduler_sim):
        """Two sync cmds create a strict order: 0 → 1 → 2."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256, "sync": True},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 256, "sync": True},
            {"cmd_id": 2, "opcode": OP_PUSH, "interface_id": 2,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)

        # Only cmd 0 should dispatch initially
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1
        assert seen["bfm1"] == 0, "Cmd 1 blocked by sync chain from cmd 0"
        assert seen["bfm2"] == 0, "Cmd 2 blocked by sync chain from cmd 0"

        # Complete cmd 0 → cmd 1 dispatches
        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0)
        seen2 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen2["bfm1"] == 1, "Cmd 1 dispatches after cmd 0 committed"
        assert seen2["bfm2"] == 0, "Cmd 2 still blocked by sync on cmd 1"

        # Complete cmd 1 → cmd 2 dispatches
        _set_bfm_done(scheduler_sim, bfm=1, cmd_id=1)
        seen3 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen3["bfm2"] == 1, "Cmd 2 dispatches after cmd 1 committed"


# ═══════════════════════════════════════════════════════════════════════════
# Barrier Fence Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBarrierFence:
    """Test BARRIER opcode: blocks until ALL prior commands are committed."""

    def test_barrier_blocks_until_all_committed(self, scheduler_sim):
        """BARRIER (cmd 2) does not commit until cmds 0 and 1 are committed."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 2, "opcode": OP_BARRIER, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 0},
            {"cmd_id": 3, "opcode": OP_PUSH, "interface_id": 2,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)

        # Cmds 0,1 dispatch. Barrier blocks cmd 3.
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm0"] == 1 and seen["bfm1"] == 1
        assert seen["bfm2"] == 0, "Cmd 3 blocked by barrier"

        # Complete cmd 0 only — barrier still blocked
        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0)
        seen2 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen2["bfm2"] == 0, "Cmd 3 still blocked — barrier not committed yet"

        # Complete cmd 1 — barrier can now commit, then cmd 3 dispatches
        _set_bfm_done(scheduler_sim, bfm=1, cmd_id=1)
        seen3 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen3["bfm2"] == 1, "Cmd 3 dispatches after barrier clears"

    def test_barrier_self_commits(self, scheduler_sim):
        """BARRIER with no prior cmds immediately self-commits."""
        commands = [
            {"cmd_id": 0, "opcode": OP_BARRIER, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 0},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=5)
        assert seen["bfm0"] == 1, "Cmd 1 dispatches after barrier self-commits"

    def test_store_self_commits(self, scheduler_sim):
        """STORE opcode self-commits like BARRIER."""
        commands = [
            {"cmd_id": 0, "opcode": OP_STORE, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 0},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=5)
        assert seen["bfm0"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# LOAD Pre-committed
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadPreCommitted:
    """Test LOAD opcode: pre-committed at batch start (no BFM dispatch)."""

    def test_load_pre_committed(self, scheduler_sim):
        """LOAD cmd is pre-committed — dependent cmd can dispatch immediately."""
        commands = [
            {"cmd_id": 0, "opcode": OP_LOAD, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256,
             "deps": [0]},  # depends on LOAD
        ]
        _setup(scheduler_sim, commands)
        seen = _collect_dispatches(scheduler_sim, ticks=5)
        assert seen["bfm0"] == 1, "Cmd 1 dispatches — LOAD dep already committed"


# ═══════════════════════════════════════════════════════════════════════════
# Error Paths
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerErrors:
    """Test scheduler error reporting."""

    def test_bfm_map_error(self, scheduler_sim):
        """Command with unmapped interface_id → ERR_UNKNOWN_OPCODE (code 6).

        NOTE: iface_to_bfm is set at create time (default: iface 0-3 → BFM 0-3,
        iface 4-7 → -1). The mapping is captured by build_bfm_map at feed_done.
        """
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 5,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        _tick(scheduler_sim, 5)

        state = _get_state(scheduler_sim)
        assert state["error_flag"] == 1, "Should flag error for unmapped BFM"
        assert state["error_cmd_id"] == 0
        # ERR_UNKNOWN_OPCODE (6) is what the scheduler reports for cmd_bfm_map < 0
        assert state["error_code"] == 6

    def test_bfm_done_with_error(self, scheduler_sim):
        """BFM reports error via done_error → scheduler propagates."""
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        _collect_dispatches(scheduler_sim, ticks=3)  # let it dispatch

        # Simulate BFM error
        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0, error=1, error_code=2)
        _tick(scheduler_sim, 2)

        state = _get_state(scheduler_sim)
        assert state["error_flag"] == 1
        assert state["error_code"] == 2  # ERR_POLL_TIMEOUT


# ═══════════════════════════════════════════════════════════════════════════
# Completion & Drain
# ═══════════════════════════════════════════════════════════════════════════


class TestCompletionDrain:
    """Test all_committed and all_drained termination signals."""

    def test_all_committed_after_all_done(self, scheduler_sim):
        """all_committed asserted after all commands complete.

        NOTE: When all BFMs are idle and all commands committed,
        all_drained fires immediately and batch_active clears. So we
        hold BFMs non-idle to observe all_committed before drain.
        """
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        # Hold BFMs non-idle so all_drained doesn't fire
        scheduler_sim._send({"cmd": "set_bfm_idle", "bfm": 0, "idle": 0})
        scheduler_sim._send({"cmd": "set_bfm_idle", "bfm": 1, "idle": 0})

        _collect_dispatches(scheduler_sim, ticks=3)

        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0)
        _set_bfm_done(scheduler_sim, bfm=1, cmd_id=1)
        r = _tick(scheduler_sim, 3)

        assert r["all_committed"] == 1
        assert r["all_drained"] == 0, "BFMs not idle — should not drain"

    def test_all_drained_requires_bfm_idle(self, scheduler_sim):
        """all_drained needs all_committed AND all BFMs idle.

        We verify the negative: with BFMs non-idle, all_drained stays 0
        even after all commands are committed. This is the complement of
        test_all_committed_after_all_done which tests the positive case.
        """
        commands = [
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256},
        ]
        _setup(scheduler_sim, commands)
        # Hold BFM 0 non-idle BEFORE dispatching to prevent premature drain
        scheduler_sim._send({"cmd": "set_bfm_idle", "bfm": 0, "idle": 0})

        _collect_dispatches(scheduler_sim, ticks=3)

        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=0)

        # Tick several cycles — all_drained should remain 0 while BFM is non-idle
        for _ in range(5):
            r = _tick(scheduler_sim, 1)
            assert r["all_drained"] == 0, "all_drained should wait for BFM idle"

        # Verify all_committed IS set though
        assert r["all_committed"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# NPU 3D-like Patterns
# ═══════════════════════════════════════════════════════════════════════════


class TestNPU3DSchedulerPatterns:
    """Test scheduler with NPU 3D-like command patterns."""

    def test_npu_layer_execution(self, scheduler_sim):
        """NPU layer: WRITE_REGs → sync → PUSH weights → PUSH ifm → BARRIER → PULL ofm."""
        commands = [
            # Register config (AXI4-Lite, BFM 3)
            {"cmd_id": 0, "opcode": OP_WRITE_REG, "interface_id": 3,
             "protocol": PROTO_AXI4L, "size": 4, "sync": True},
            # Push weights (AXI4-Stream, BFM 0)
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 27648},
            # Push ifm (AXI4-Stream, BFM 1)
            {"cmd_id": 2, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 2048},
            # Barrier — wait for all above
            {"cmd_id": 3, "opcode": OP_BARRIER, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 0},
            # Pull ofm (AXI4-Stream, BFM 2)
            {"cmd_id": 4, "opcode": OP_PULL, "interface_id": 2,
             "protocol": PROTO_AXI4S, "size": 2048},
        ]
        _setup(scheduler_sim, commands)
        # Hold all BFMs non-idle to prevent premature drain
        for b in range(4):
            scheduler_sim._send({"cmd": "set_bfm_idle", "bfm": b, "idle": 0})

        # Cmd 0 (WRITE_REG, sync) dispatches to BFM 3. Others blocked by sync.
        seen = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen["bfm3"] == 1, "WRITE_REG dispatches to BFM 3"
        assert seen["bfm0"] == 0 and seen["bfm1"] == 0, "Blocked by sync"

        # Complete WRITE_REG → cmds 1,2 dispatch concurrently
        _set_bfm_done(scheduler_sim, bfm=3, cmd_id=0)
        seen2 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen2["bfm0"] == 1 and seen2["bfm1"] == 1, \
            "Cmds 1,2 should dispatch after sync"
        assert seen2["bfm2"] == 0, "Cmd 4 blocked by barrier"

        # Complete cmds 1,2 → barrier clears → cmd 4 dispatches
        _set_bfm_done(scheduler_sim, bfm=0, cmd_id=1)
        _set_bfm_done(scheduler_sim, bfm=1, cmd_id=2)
        seen3 = _collect_dispatches(scheduler_sim, ticks=3)
        assert seen3["bfm2"] == 1, "Cmd 4 dispatches after barrier clears"

        # Complete cmd 4, then allow drain
        _set_bfm_done(scheduler_sim, bfm=2, cmd_id=4)
        r = _tick(scheduler_sim, 3)
        assert r["all_committed"] == 1
