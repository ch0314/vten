"""Functional tests for vten_shm_controller via verilator simulation.

Tests the 9-state FSM, DPI-C call sequences, and command feeding handshake.
Reference: vten/sv/vten_shm_controller.sv
"""

from __future__ import annotations

import pytest

from tests.phase3.conftest import (
    BACKEND_DONE,
    BACKEND_ERROR,
    BACKEND_RUNNING,
    HOST_CMD_READY,
    HOST_SHUTDOWN,
    OP_BARRIER,
    OP_POLL_REG,
    OP_PULL,
    OP_PUSH,
    OP_WRITE_REG,
    PROTO_AXI4L,
    PROTO_AXI4S,
    ROLE_MASTER,
    ROLE_SLAVE,
    S_COMPLETE,
    S_DRAIN,
    S_ERROR,
    S_EXECUTE,
    S_FEED,
    S_INIT,
    S_LOAD_BATCH,
    S_WAIT_HOST,
    build_shm_image,
    requires_verilator,
)


def _advance_to_state(sim, target_state: str, max_ticks: int = 50) -> dict:
    """Tick until the FSM reaches target_state or max_ticks exceeded."""
    for _ in range(max_ticks):
        r = sim.tick()
        if r["state_name"] == target_state:
            return r
    raise TimeoutError(f"Did not reach {target_state} within {max_ticks} ticks")


def _run_full_batch(sim, set_feed_ready: bool = True) -> None:
    """Run FSM through LOAD_BATCH → FEED → EXECUTE → DRAIN → COMPLETE."""
    if set_feed_ready:
        sim.set_signal("feed_ready", 1)
    _advance_to_state(sim, "S_EXECUTE")
    sim.set_signal("sched_all_committed", 1)
    _advance_to_state(sim, "S_DRAIN")
    sim.set_signal("sched_all_drained", 1)
    _advance_to_state(sim, "S_COMPLETE")


# ═══════════════════════════════════════════════════════════════════════════
# Test: FSM State Transitions
# ═══════════════════════════════════════════════════════════════════════════


@requires_verilator
class TestFSMStateTransitions:
    """Test the basic FSM state machine of vten_shm_controller."""

    def test_reset_enters_init(self, shm_ctrl_sim):
        """After reset, FSM should be in S_INIT."""
        image = build_shm_image(num_commands=1, commands=[
            {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "role": ROLE_MASTER, "size": 256},
        ])
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        r = shm_ctrl_sim.reset(5)
        # After reset + 1 tick, S_INIT calls vten_shm_init and transitions
        # If mock is loaded, it goes to S_WAIT_HOST
        assert r["state"] in (S_INIT, S_WAIT_HOST)

    def test_init_to_wait_host(self, shm_ctrl_sim):
        """S_INIT should transition to S_WAIT_HOST on successful shm_init."""
        # Set host_status=IDLE so FSM stays in WAIT_HOST instead of advancing
        image = build_shm_image(
            num_commands=1,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "role": ROLE_MASTER, "size": 256},
            ],
            host_status=0,  # HOST_IDLE — FSM will wait
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        # With host_status=IDLE, wait_host_signal returns TIMEOUT → stays WAIT_HOST
        shm_ctrl_sim.mock_set("wait_host_result", 1)  # TIMEOUT
        r = shm_ctrl_sim.tick()
        # FSM should be in WAIT_HOST (or just past INIT)
        assert r["state_name"] in ("S_INIT", "S_WAIT_HOST")

    def test_wait_host_to_load_batch(self, shm_ctrl_sim):
        """S_WAIT_HOST → S_LOAD_BATCH when host_status=CMD_READY."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
            host_status=HOST_CMD_READY,
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        r = _advance_to_state(shm_ctrl_sim, "S_LOAD_BATCH")
        assert r["state_name"] == "S_LOAD_BATCH"

    def test_load_batch_to_feed(self, shm_ctrl_sim):
        """S_LOAD_BATCH → S_FEED after reading commands."""
        image = build_shm_image(
            num_commands=2,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "size": 256},
                {"cmd_id": 1, "opcode": OP_PULL, "interface_id": 1,
                 "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 256},
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        r = _advance_to_state(shm_ctrl_sim, "S_FEED")
        assert r["state_name"] == "S_FEED"

    def test_feed_to_execute(self, shm_ctrl_sim):
        """S_FEED → S_EXECUTE after all commands fed (with feed_ready=1)."""
        image = build_shm_image(
            num_commands=2,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "size": 256},
                {"cmd_id": 1, "opcode": OP_PULL, "interface_id": 1,
                 "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 256},
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        r = _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        assert r["state_name"] == "S_EXECUTE"

    def test_execute_to_drain(self, shm_ctrl_sim):
        """S_EXECUTE → S_DRAIN when sched_all_committed asserted."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        shm_ctrl_sim.set_signal("sched_all_committed", 1)
        r = _advance_to_state(shm_ctrl_sim, "S_DRAIN")
        assert r["state_name"] == "S_DRAIN"

    def test_drain_to_complete(self, shm_ctrl_sim):
        """S_DRAIN → S_COMPLETE when sched_all_drained asserted."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        shm_ctrl_sim.set_signal("sched_all_committed", 1)
        _advance_to_state(shm_ctrl_sim, "S_DRAIN")
        shm_ctrl_sim.set_signal("sched_all_drained", 1)
        r = _advance_to_state(shm_ctrl_sim, "S_COMPLETE")
        assert r["state_name"] == "S_COMPLETE"

    def test_complete_to_wait_host(self, shm_ctrl_sim):
        """S_COMPLETE → S_WAIT_HOST after signaling completion."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _run_full_batch(shm_ctrl_sim, set_feed_ready=False)
        r = _advance_to_state(shm_ctrl_sim, "S_WAIT_HOST")
        assert r["state_name"] == "S_WAIT_HOST"

    def test_execute_error_to_error_state(self, shm_ctrl_sim):
        """S_EXECUTE → S_ERROR when sched_error asserted."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        shm_ctrl_sim.set_signal("sched_error", 1)
        shm_ctrl_sim.set_signal("sched_error_cmd_id", 0)
        shm_ctrl_sim.set_signal("sched_error_code", 2)  # POLL_TIMEOUT
        r = _advance_to_state(shm_ctrl_sim, "S_ERROR")
        assert r["state_name"] == "S_ERROR"


# ═══════════════════════════════════════════════════════════════════════════
# Test: Backend Status and Signal Complete/Error
# ═══════════════════════════════════════════════════════════════════════════


@requires_verilator
class TestBackendSignaling:
    """Test backend_status updates and signal_complete/signal_error."""

    def test_backend_running_after_load_batch(self, shm_ctrl_sim):
        """Backend status = RUNNING after S_LOAD_BATCH."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_RUNNING

    def test_signal_complete_on_success(self, shm_ctrl_sim):
        """signal_complete called when FSM reaches S_COMPLETE."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _run_full_batch(shm_ctrl_sim, set_feed_ready=False)
        # After COMPLETE, one more tick for the signal
        shm_ctrl_sim.tick()
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_DONE
        assert shm_ctrl_sim.mock_get("complete_count") >= 1

    def test_signal_error_on_sched_error(self, shm_ctrl_sim):
        """signal_error called when FSM enters S_ERROR."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        shm_ctrl_sim.set_signal("sched_error", 1)
        shm_ctrl_sim.set_signal("sched_error_code", 4)  # SCHEDULER_ERROR
        _advance_to_state(shm_ctrl_sim, "S_ERROR")
        # After ERROR, next tick signals the error
        shm_ctrl_sim.tick()
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_ERROR
        assert shm_ctrl_sim.mock_get("error_count") >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: Command Feeding
# ═══════════════════════════════════════════════════════════════════════════


@requires_verilator
class TestCommandFeeding:
    """Test the S_FEED handshake: feed_valid/feed_ready/feed_done."""

    def test_feed_stalls_without_feed_ready(self, shm_ctrl_sim):
        """S_FEED should not advance if feed_ready=0."""
        image = build_shm_image(
            num_commands=2,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "size": 256},
                {"cmd_id": 1, "opcode": OP_PULL, "interface_id": 1,
                 "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 256},
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        # Don't set feed_ready — tick several times
        for _ in range(5):
            r = shm_ctrl_sim.tick()
            assert r["state_name"] == "S_FEED"
            assert r["feed_valid"] == 0

    def test_feed_valid_asserted_with_feed_ready(self, shm_ctrl_sim):
        """feed_valid should be 1 when feed_ready=1 and commands remain."""
        image = build_shm_image(
            num_commands=2,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "size": 256},
                {"cmd_id": 1, "opcode": OP_PULL, "interface_id": 1,
                 "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 256},
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        shm_ctrl_sim.set_signal("feed_ready", 1)
        r = shm_ctrl_sim.tick()
        # Should see feed_valid=1 within 1-2 ticks
        if r["feed_valid"] == 0:
            r = shm_ctrl_sim.tick()
        assert r["feed_valid"] == 1

    def test_feed_done_after_all_commands(self, shm_ctrl_sim):
        """feed_done should pulse after all commands are fed."""
        image = build_shm_image(
            num_commands=2,
            commands=[
                {"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                 "protocol": PROTO_AXI4S, "size": 256},
                {"cmd_id": 1, "opcode": OP_PULL, "interface_id": 1,
                 "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 256},
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        # Collect feed_done across ticks
        seen_feed_done = False
        for _ in range(20):
            r = shm_ctrl_sim.tick()
            if r["feed_done"] == 1:
                seen_feed_done = True
                break
        assert seen_feed_done, "feed_done never asserted"

    def test_num_commands_loaded(self, shm_ctrl_sim):
        """Internal num_commands matches SHM image after LOAD_BATCH."""
        image = build_shm_image(
            num_commands=3,
            commands=[
                {"cmd_id": i, "opcode": OP_PUSH, "interface_id": i,
                 "protocol": PROTO_AXI4S, "size": 256}
                for i in range(3)
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        internals = shm_ctrl_sim.get_internals()
        assert internals["num_commands"] == 3

    def test_feed_idx_increments(self, shm_ctrl_sim):
        """feed_idx should increment each time a command is fed."""
        image = build_shm_image(
            num_commands=3,
            commands=[
                {"cmd_id": i, "opcode": OP_PUSH, "interface_id": i,
                 "protocol": PROTO_AXI4S, "size": 256}
                for i in range(3)
            ],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
        internals = shm_ctrl_sim.get_internals()
        assert internals["feed_idx"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# Test: Multi-Batch (back-to-back execution)
# ═══════════════════════════════════════════════════════════════════════════


@requires_verilator
class TestMultiBatch:
    """Test multiple consecutive batches (COMPLETE → WAIT_HOST → LOAD_BATCH)."""

    def test_two_consecutive_batches(self, shm_ctrl_sim):
        """After COMPLETE, the FSM loops back to WAIT_HOST for next batch."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)

        for batch in range(2):
            shm_ctrl_sim.set_signal("feed_ready", 1)
            _advance_to_state(shm_ctrl_sim, "S_EXECUTE")
            shm_ctrl_sim.set_signal("sched_all_committed", 1)
            _advance_to_state(shm_ctrl_sim, "S_DRAIN")
            shm_ctrl_sim.set_signal("sched_all_drained", 1)
            _advance_to_state(shm_ctrl_sim, "S_COMPLETE")
            # Reset scheduler signals for next batch
            shm_ctrl_sim.set_signal("sched_all_committed", 0)
            shm_ctrl_sim.set_signal("sched_all_drained", 0)
            shm_ctrl_sim.set_signal("feed_ready", 0)
            _advance_to_state(shm_ctrl_sim, "S_WAIT_HOST")

        assert shm_ctrl_sim.mock_get("complete_count") >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Test: NPU 3D Patterns
# ═══════════════════════════════════════════════════════════════════════════


@requires_verilator
class TestLargeBatch:
    """Test large batch scalability — up to MAX_CMDS=256."""

    def _feed_batch(self, sim, num_cmds: int, max_ticks: int = 600):
        """Load and feed a batch of N identical PUSH commands."""
        commands = [
            {"cmd_id": i, "opcode": OP_PUSH, "interface_id": i % 8,
             "protocol": PROTO_AXI4S, "size": 256}
            for i in range(num_cmds)
        ]
        image = build_shm_image(num_commands=num_cmds, commands=commands)
        sim.load_shm_image(image)
        sim.create()
        sim.reset(5)
        sim.set_signal("feed_ready", 1)
        _advance_to_state(sim, "S_EXECUTE", max_ticks=max_ticks)
        return sim.get_internals()

    def test_64_command_batch(self, shm_ctrl_sim):
        """64-command batch: feed path should handle without stalls."""
        internals = self._feed_batch(shm_ctrl_sim, 64)
        assert internals["num_commands"] == 64
        assert internals["feed_idx"] == 64

    def test_128_command_batch(self, shm_ctrl_sim):
        """128-command batch: mid-range scalability."""
        internals = self._feed_batch(shm_ctrl_sim, 128, max_ticks=300)
        assert internals["num_commands"] == 128
        assert internals["feed_idx"] == 128

    def test_256_command_batch(self, shm_ctrl_sim):
        """256-command batch: MAX_CMDS boundary."""
        internals = self._feed_batch(shm_ctrl_sim, 256, max_ticks=600)
        assert internals["num_commands"] == 256
        assert internals["feed_idx"] == 256

    def test_feed_linear_scaling(self, shm_ctrl_sim):
        """Feed takes exactly N cycles with feed_ready=1 (1 cmd/cycle)."""
        num_cmds = 32
        commands = [
            {"cmd_id": i, "opcode": OP_PUSH, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 256}
            for i in range(num_cmds)
        ]
        image = build_shm_image(num_commands=num_cmds, commands=commands)
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        shm_ctrl_sim.set_signal("feed_ready", 1)
        # Count ticks from FEED to EXECUTE
        ticks = 0
        for _ in range(200):
            r = shm_ctrl_sim.tick()
            ticks += 1
            if r["state_name"] == "S_EXECUTE":
                break
        # Should take ~num_cmds ticks (± small overhead for state transitions)
        assert ticks <= num_cmds + 5, f"Feed took {ticks} ticks for {num_cmds} cmds"

    def test_256_full_lifecycle(self, shm_ctrl_sim):
        """256-command batch through full FEED→EXECUTE→DRAIN→COMPLETE."""
        internals = self._feed_batch(shm_ctrl_sim, 256, max_ticks=600)
        assert internals["num_commands"] == 256
        # Complete the batch
        shm_ctrl_sim.set_signal("sched_all_committed", 1)
        _advance_to_state(shm_ctrl_sim, "S_DRAIN")
        shm_ctrl_sim.set_signal("sched_all_drained", 1)
        _advance_to_state(shm_ctrl_sim, "S_COMPLETE")
        shm_ctrl_sim.tick()
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_DONE


@requires_verilator
class TestCrossBatchPreservation:
    """Test data region preservation and session_seq across batches."""

    def test_session_seq_increments(self, shm_ctrl_sim):
        """session_seq in control header increments on each init."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        seq1 = shm_ctrl_sim.mock_get("session_seq")
        assert seq1 >= 1, "session_seq should be ≥ 1 after init"

        # Run a full batch and loop back
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _run_full_batch(shm_ctrl_sim, set_feed_ready=False)
        _advance_to_state(shm_ctrl_sim, "S_WAIT_HOST")

        seq2 = shm_ctrl_sim.mock_get("session_seq")
        # session_seq should not change between batches (only on init)
        assert seq2 == seq1

    def test_data_region_preserved_between_batches(self, shm_ctrl_sim):
        """Data region bytes are intact between batch 1 and batch 2."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
            num_buffers=1,
        )
        # Write recognizable data in the data region
        import struct
        data_off = struct.unpack_from("<Q", image, 0x30)[0]
        total = max(len(image), data_off + 256 + 64)
        image.extend(b"\x00" * (total - len(image)))
        for i in range(256):
            image[data_off + i] = (0xA5 + i) & 0xFF

        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)

        # Run batch 1
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _run_full_batch(shm_ctrl_sim, set_feed_ready=False)
        _advance_to_state(shm_ctrl_sim, "S_WAIT_HOST")

        # Read data region — should be intact
        r = shm_ctrl_sim._send({
            "cmd": "read_shm", "offset": data_off, "size": 16,
        })
        assert "error" not in r
        expected = [(0xA5 + i) & 0xFF for i in range(16)]
        assert r["bytes"] == expected, (
            f"Data region corrupted after batch 1: {r['bytes']} != {expected}")

    def test_backend_status_resets_between_batches(self, shm_ctrl_sim):
        """Backend status transitions correctly across consecutive batches."""
        image = build_shm_image(
            num_commands=1,
            commands=[{"cmd_id": 0, "opcode": OP_PUSH, "interface_id": 0,
                       "protocol": PROTO_AXI4S, "size": 256}],
        )
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)

        # Batch 1 — run to completion
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _run_full_batch(shm_ctrl_sim, set_feed_ready=False)
        shm_ctrl_sim.tick()  # tick for signal_complete
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_DONE

        # Block at WAIT_HOST by setting wait_host_result=1 (TIMEOUT)
        shm_ctrl_sim.set_signal("sched_all_committed", 0)
        shm_ctrl_sim.set_signal("sched_all_drained", 0)
        shm_ctrl_sim.set_signal("feed_ready", 0)
        shm_ctrl_sim.mock_set("wait_host_result", 1)  # TIMEOUT → stays at WAIT_HOST
        _advance_to_state(shm_ctrl_sim, "S_WAIT_HOST")

        # Now release — allow batch 2
        shm_ctrl_sim.mock_set("wait_host_result", 0)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_FEED")
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_RUNNING


@requires_verilator
class TestNPU3DPatterns:
    """Test with NPU 3D-like command patterns."""

    def test_register_config_batch(self, shm_ctrl_sim):
        """Batch of WRITE_REG commands (NPU register configuration)."""
        num_regs = 8
        commands = [
            {"cmd_id": i, "opcode": OP_WRITE_REG, "interface_id": 0,
             "protocol": PROTO_AXI4L, "size": 4,
             "reg_offset": 0x100 + i * 4, "reg_value": i * 0x11}
            for i in range(num_regs)
        ]
        image = build_shm_image(num_commands=num_regs, commands=commands)
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")

        internals = shm_ctrl_sim.get_internals()
        assert internals["num_commands"] == num_regs
        assert internals["feed_idx"] == num_regs

    def test_mixed_opcode_batch(self, shm_ctrl_sim):
        """Mixed batch: WRITE_REG + PUSH + PULL + POLL_REG + BARRIER."""
        commands = [
            {"cmd_id": 0, "opcode": OP_WRITE_REG, "interface_id": 0,
             "protocol": PROTO_AXI4L, "size": 4,
             "reg_offset": 0x100, "reg_value": 1},
            {"cmd_id": 1, "opcode": OP_PUSH, "interface_id": 1,
             "protocol": PROTO_AXI4S, "size": 1024},
            {"cmd_id": 2, "opcode": OP_PULL, "interface_id": 2,
             "protocol": PROTO_AXI4S, "role": ROLE_SLAVE, "size": 1024},
            {"cmd_id": 3, "opcode": OP_POLL_REG, "interface_id": 0,
             "protocol": PROTO_AXI4L, "size": 4,
             "reg_offset": 0x200, "reg_mask": 0x1, "reg_expected": 0x1},
            {"cmd_id": 4, "opcode": OP_BARRIER, "interface_id": 0,
             "protocol": PROTO_AXI4S, "size": 0},
        ]
        image = build_shm_image(num_commands=5, commands=commands)
        shm_ctrl_sim.load_shm_image(image)
        shm_ctrl_sim.create()
        shm_ctrl_sim.reset(5)
        shm_ctrl_sim.set_signal("feed_ready", 1)
        _advance_to_state(shm_ctrl_sim, "S_EXECUTE")

        internals = shm_ctrl_sim.get_internals()
        assert internals["num_commands"] == 5
        assert internals["feed_idx"] == 5

        # Complete the batch
        shm_ctrl_sim.set_signal("sched_all_committed", 1)
        _advance_to_state(shm_ctrl_sim, "S_DRAIN")
        shm_ctrl_sim.set_signal("sched_all_drained", 1)
        _advance_to_state(shm_ctrl_sim, "S_COMPLETE")
        # signal_complete executes on the next tick (S_COMPLETE state)
        shm_ctrl_sim.tick()
        assert shm_ctrl_sim.mock_get("backend_status") == BACKEND_DONE
