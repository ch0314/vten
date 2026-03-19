"""Functional tests for vten_bfm_axi4s — AXI4-Stream BFM (MASTER mode).

Tests verify that the BFM correctly reads data from SHM via DPI-C mock,
drives AXI4-Stream transactions (tdata/tvalid/tlast), and signals completion.
The C++ driver acts as the stream slave (always tready=1).
"""

from __future__ import annotations

import struct

from tests.phase3.conftest import (
    OP_PUSH,
    PROTO_AXI4S,
    build_shm_image,
    requires_verilator,
)

pytestmark = requires_verilator

BYTES_PER_BEAT = 32  # DATA_W=256 → 32 bytes/beat


# ── Helpers ──

def _build_push_image(num_beats: int, buf_id: int = 0, cmd_id: int = 0):
    """Build SHM image with a single PUSH command and data region."""
    size = num_beats * BYTES_PER_BEAT
    num_cmds = cmd_id + 1  # Allocate enough slots for the cmd_id
    image = build_shm_image(
        num_commands=num_cmds,
        commands=[{
            "opcode": OP_PUSH, "cmd_id": cmd_id, "interface_id": 0,
            "protocol": PROTO_AXI4S, "role": 0, "buffer_id": buf_id,
            "size": size,
        }],
        num_buffers=1,
    )
    # Extend image with data region
    data_off = struct.unpack_from("<Q", image, 0x30)[0]
    total = max(len(image), data_off + size + 64)
    image.extend(b"\x00" * (total - len(image)))
    # Write recognizable pattern
    for i in range(size):
        image[data_off + i] = i & 0xFF
    return image


def _setup(sim, image):
    """Load SHM image, create DUT, and reset."""
    sim.load_shm_image(image)
    sim.create()
    sim.reset(5)


def _issue_push(sim, cmd_id=0, buf_id=0, size=64):
    """Issue a PUSH command."""
    return sim._send({
        "cmd": "issue_cmd", "opcode": OP_PUSH,
        "cmd_id": cmd_id, "buffer_id": buf_id, "size": size,
    })


def _run_until_done(sim, max_ticks=200):
    return sim._send({"cmd": "run_until_done", "max_ticks": max_ticks})


def _get_debug(sim):
    return sim._send({"cmd": "get_debug"})


def _tick(sim, n=1):
    return sim._send({"cmd": "tick", "n": n})


# ══════════════════════════════════════════════════════════════════════════════
# Command Completion Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCommandCompletion:
    """Test PUSH command completion signaling."""

    def test_single_beat_push(self, axi4s_sim):
        """PUSH with 1 beat (32 bytes) completes."""
        image = _build_push_image(num_beats=1)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["done_error"] == 0
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == 1
        assert dbg["expected_beats"] == 1

    def test_two_beat_push(self, axi4s_sim):
        """PUSH with 2 beats (64 bytes) completes."""
        image = _build_push_image(num_beats=2)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=2 * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["done_error"] == 0
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == 2

    def test_four_beat_push(self, axi4s_sim):
        """PUSH with 4 beats (128 bytes)."""
        image = _build_push_image(num_beats=4)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=4 * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["done_error"] == 0
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == 4

    def test_large_push(self, axi4s_sim):
        """PUSH with 16 beats (512 bytes)."""
        image = _build_push_image(num_beats=16)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=16 * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim, max_ticks=100)
        assert r["done"] == 1
        assert r["done_error"] == 0
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == 16

    def test_done_cmd_id(self, axi4s_sim):
        """done_cmd_id matches the issued command."""
        image = _build_push_image(num_beats=2, cmd_id=42)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, cmd_id=42, size=2 * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["done_cmd_id"] == 42


# ══════════════════════════════════════════════════════════════════════════════
# Stream Signal Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStreamSignals:
    """Test AXI4-Stream signal behavior."""

    def test_tlast_seen(self, axi4s_sim):
        """tlast is asserted during the last beat."""
        image = _build_push_image(num_beats=4)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=4 * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["tlast_seen"] == 1

    def test_tlast_single_beat(self, axi4s_sim):
        """tlast is asserted for single-beat transfer."""
        image = _build_push_image(num_beats=1)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim)
        assert r["done"] == 1
        assert r["tlast_seen"] == 1

    def test_valid_cycles_match_beats(self, axi4s_sim):
        """Number of tvalid cycles is at least the expected beat count."""
        for num_beats in [1, 2, 4, 8]:
            image = _build_push_image(num_beats=num_beats)
            _setup(axi4s_sim, image)
            _issue_push(axi4s_sim, size=num_beats * BYTES_PER_BEAT)
            r = _run_until_done(axi4s_sim, max_ticks=100)
            assert r["done"] == 1, f"Failed for {num_beats} beats"
            assert r["valid_cycles"] >= num_beats


# ══════════════════════════════════════════════════════════════════════════════
# Idle & Reset Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIdleAndReset:
    """Test idle signaling and reset behavior."""

    def test_idle_after_reset(self, axi4s_sim):
        """BFM is idle after reset."""
        image = _build_push_image(num_beats=1)
        _setup(axi4s_sim, image)
        r = _tick(axi4s_sim, 1)
        assert r["idle"] == 1

    def test_idle_after_completion(self, axi4s_sim):
        """BFM returns to idle after command completes."""
        image = _build_push_image(num_beats=2)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=2 * BYTES_PER_BEAT)
        _run_until_done(axi4s_sim)
        r = _tick(axi4s_sim, 3)
        assert r["idle"] == 1

    def test_not_idle_during_transfer(self, axi4s_sim):
        """BFM is not idle while processing command."""
        image = _build_push_image(num_beats=8)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=8 * BYTES_PER_BEAT)
        r = _tick(axi4s_sim, 3)
        assert r["idle"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# NPU 3D Patterns
# ══════════════════════════════════════════════════════════════════════════════


class TestNPU3DPatterns:
    """Test patterns matching real NPU 3D data transfers."""

    def test_npu_weight_tile_push(self, axi4s_sim):
        """Push a weight tile: Ti*To*K^3 = 32*32*27 = 27648 bytes = 864 beats."""
        num_beats = 864
        size = num_beats * BYTES_PER_BEAT
        image = build_shm_image(
            num_commands=1,
            commands=[{
                "opcode": OP_PUSH, "cmd_id": 0, "interface_id": 0,
                "protocol": PROTO_AXI4S, "role": 0, "buffer_id": 0,
                "size": size,
            }],
            num_buffers=1,
        )
        data_off = struct.unpack_from("<Q", image, 0x30)[0]
        total = max(len(image), data_off + size + 64)
        image.extend(b"\x00" * (total - len(image)))
        for i in range(size):
            image[data_off + i] = i & 0xFF

        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=size)
        r = _run_until_done(axi4s_sim, max_ticks=2000)
        assert r["done"] == 1
        assert r["done_error"] == 0
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == num_beats

    def test_npu_feature_map_push(self, axi4s_sim):
        """Push feature map tile: D*H*W*Ti = 1*8*8*32 = 2048 bytes = 64 beats."""
        num_beats = 64
        image = _build_push_image(num_beats=num_beats)
        _setup(axi4s_sim, image)
        _issue_push(axi4s_sim, size=num_beats * BYTES_PER_BEAT)
        r = _run_until_done(axi4s_sim, max_ticks=500)
        assert r["done"] == 1
        assert r["done_error"] == 0
        assert r["tlast_seen"] == 1
        dbg = _get_debug(axi4s_sim)
        assert dbg["beat_count"] == num_beats
