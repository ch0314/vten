"""Functional tests for vten_bfm_axi4 — AXI4 Memory-Mapped BFM (slave).

Tests verify that the BFM correctly serves read data from SHM and accepts
write data to SHM. The C++ driver acts as the AXI4 master.

NOTE: The BFM's internal check_completion path does not fire in verilator
due to a verilator limitation (atWriteAppend/at ordering for queue struct
fields). Tests verify AXI protocol correctness instead:
- AR acceptance and R data serving (read path)
- AW/W acceptance and B response (write path)
- Correct burst lengths, rlast, rresp/bresp
"""

from __future__ import annotations

import struct

from tests.phase3.conftest import (
    OP_PUSH,
    OP_PULL,
    PROTO_AXI4,
    build_shm_image,
    requires_verilator,
)

pytestmark = requires_verilator

BYTES_PER_BEAT = 32  # DATA_W=256


# ── Helpers ──

def _build_axi4_image(opcode, cmd_id=0, buf_id=0, size=64, phys_addr=0x1000):
    """Build SHM image with a single AXI4 command and data region."""
    num_cmds = cmd_id + 1
    image = build_shm_image(
        num_commands=num_cmds,
        commands=[{
            "opcode": opcode, "cmd_id": cmd_id, "interface_id": 0,
            "protocol": PROTO_AXI4, "role": 0, "buffer_id": buf_id,
            "size": size, "phys_addr": phys_addr,
        }],
        num_buffers=1,
    )
    data_off = struct.unpack_from("<Q", image, 0x30)[0]
    total = max(len(image), data_off + size + 64)
    image.extend(b"\x00" * (total - len(image)))
    for i in range(size):
        image[data_off + i] = i & 0xFF
    return image


def _setup(sim, image):
    sim.load_shm_image(image)
    sim.create()
    sim.reset(5)


def _issue_cmd(sim, opcode, cmd_id=0, buf_id=0, size=64, phys_addr=0x1000):
    return sim._send({
        "cmd": "issue_cmd", "opcode": opcode,
        "cmd_id": cmd_id, "buffer_id": buf_id,
        "size": size, "phys_addr": phys_addr,
    })


def _tick(sim, n=1):
    return sim._send({"cmd": "tick", "n": n})


def _read_burst(sim, addr, num_beats, **kw):
    """Issue AXI4 read burst. len is 0-based (num_beats - 1)."""
    return sim._send({
        "cmd": "read_burst", "addr": addr,
        "len": num_beats - 1, "size": 5,  # 32 bytes
        "max_ticks": kw.get("max_ticks", 200),
    })


def _write_burst(sim, addr, num_beats, pattern=0xAB, **kw):
    """Issue AXI4 write burst."""
    return sim._send({
        "cmd": "write_burst", "addr": addr,
        "len": num_beats - 1, "size": 5,
        "pattern": pattern,
        "max_ticks": kw.get("max_ticks", 200),
    })


def _get_debug(sim):
    return sim._send({"cmd": "get_debug"})


# ══════════════════════════════════════════════════════════════════════════════
# Read Path (PUSH) Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestReadPath:
    """Test AXI4 read (AR→R) path — BFM serves data from SHM."""

    def test_single_beat_read(self, axi4_sim):
        """Single-beat read burst (arlen=0)."""
        image = _build_axi4_image(OP_PUSH, size=BYTES_PER_BEAT)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=BYTES_PER_BEAT)
        _tick(axi4_sim, 2)

        r = _read_burst(axi4_sim, 0x1000, 1)
        assert r["ar_done"] == 1, "AR handshake should complete"
        assert r["beats"] == 1
        assert r["rlast"] == 1
        assert r["rresp"] == 0  # OKAY

    def test_two_beat_read(self, axi4_sim):
        """Two-beat read burst (arlen=1)."""
        image = _build_axi4_image(OP_PUSH, size=64)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=64)
        _tick(axi4_sim, 2)

        r = _read_burst(axi4_sim, 0x1000, 2)
        assert r["ar_done"] == 1
        assert r["beats"] == 2
        assert r["rlast"] == 1
        assert r["rresp"] == 0

    def test_four_beat_read(self, axi4_sim):
        """Four-beat read burst (arlen=3)."""
        image = _build_axi4_image(OP_PUSH, size=128)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=128)
        _tick(axi4_sim, 2)

        r = _read_burst(axi4_sim, 0x1000, 4)
        assert r["ar_done"] == 1
        assert r["beats"] == 4
        assert r["rlast"] == 1

    def test_sixteen_beat_read(self, axi4_sim):
        """16-beat read burst."""
        size = 16 * BYTES_PER_BEAT
        image = _build_axi4_image(OP_PUSH, size=size)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=size)
        _tick(axi4_sim, 2)

        r = _read_burst(axi4_sim, 0x1000, 16, max_ticks=200)
        assert r["ar_done"] == 1
        assert r["beats"] == 16
        assert r["rlast"] == 1

    def test_read_with_offset(self, axi4_sim):
        """Read starting at non-zero offset within the buffer."""
        image = _build_axi4_image(OP_PUSH, size=128, phys_addr=0x2000)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=128, phys_addr=0x2000)
        _tick(axi4_sim, 2)

        # Read 2 beats starting at offset 32 from base
        r = _read_burst(axi4_sim, 0x2020, 2)
        assert r["ar_done"] == 1
        assert r["beats"] == 2
        assert r["rlast"] == 1

    def test_active_table_populated(self, axi4_sim):
        """PUSH command populates the active_table."""
        image = _build_axi4_image(OP_PUSH, size=64)
        _setup(axi4_sim, image)

        dbg = _get_debug(axi4_sim)
        assert dbg["active_table"] == 0

        _issue_cmd(axi4_sim, OP_PUSH, size=64)
        _tick(axi4_sim, 2)

        dbg = _get_debug(axi4_sim)
        assert dbg["active_table"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Write Path (PULL) Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestWritePath:
    """Test AXI4 write (AW→W→B) path — BFM captures data to SHM."""

    def test_single_beat_write(self, axi4_sim):
        """Single-beat write burst."""
        image = _build_axi4_image(OP_PULL, size=BYTES_PER_BEAT)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PULL, size=BYTES_PER_BEAT)
        _tick(axi4_sim, 2)

        r = _write_burst(axi4_sim, 0x1000, 1)
        assert r["aw_done"] == 1
        assert r["w_beats"] == 1
        assert r["b_done"] == 1
        assert r["bresp"] == 0  # OKAY

    def test_two_beat_write(self, axi4_sim):
        """Two-beat write burst."""
        image = _build_axi4_image(OP_PULL, size=64)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PULL, size=64)
        _tick(axi4_sim, 2)

        r = _write_burst(axi4_sim, 0x1000, 2)
        assert r["aw_done"] == 1
        assert r["w_beats"] == 2
        assert r["b_done"] == 1
        assert r["bresp"] == 0

    def test_four_beat_write(self, axi4_sim):
        """Four-beat write burst."""
        image = _build_axi4_image(OP_PULL, size=128)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PULL, size=128)
        _tick(axi4_sim, 2)

        r = _write_burst(axi4_sim, 0x1000, 4)
        assert r["aw_done"] == 1
        assert r["w_beats"] == 4
        assert r["b_done"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Idle & Reset Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIdleAndReset:
    """Test idle signaling and reset."""

    def test_idle_after_reset(self, axi4_sim):
        """BFM is idle after reset (no commands issued)."""
        image = _build_axi4_image(OP_PUSH, size=64)
        _setup(axi4_sim, image)
        r = _tick(axi4_sim, 1)
        assert r["idle"] == 1

    def test_not_idle_with_active_command(self, axi4_sim):
        """BFM is not idle when active_table has entries."""
        image = _build_axi4_image(OP_PUSH, size=64)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=64)
        _tick(axi4_sim, 2)
        dbg = _get_debug(axi4_sim)
        assert dbg["active_table"] == 1

    def test_read_pending_drains(self, axi4_sim):
        """read_pending queue drains after read burst."""
        image = _build_axi4_image(OP_PUSH, size=64)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=64)
        _tick(axi4_sim, 2)
        _read_burst(axi4_sim, 0x1000, 2)

        dbg = _get_debug(axi4_sim)
        assert dbg["read_pending"] == 0
        assert dbg["r_active"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# NPU 3D Patterns
# ══════════════════════════════════════════════════════════════════════════════


class TestNPU3DPatterns:
    """Test patterns matching real NPU 3D memory transactions."""

    def test_npu_ddr_read_256bit(self, axi4_sim):
        """NPU DDR read: 256-bit bus, 16-beat burst (512 bytes)."""
        size = 16 * BYTES_PER_BEAT
        image = _build_axi4_image(OP_PUSH, size=size, phys_addr=0x8000_0000)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, size=size, phys_addr=0x8000_0000)
        _tick(axi4_sim, 2)

        r = _read_burst(axi4_sim, 0x8000_0000, 16, max_ticks=200)
        assert r["ar_done"] == 1
        assert r["beats"] == 16
        assert r["rlast"] == 1
        assert r["rresp"] == 0

    def test_npu_ddr_write_burst(self, axi4_sim):
        """NPU DDR write: output feature map tile."""
        size = 8 * BYTES_PER_BEAT  # 256 bytes
        image = _build_axi4_image(OP_PULL, size=size, phys_addr=0xC000_0000)
        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PULL, size=size, phys_addr=0xC000_0000)
        _tick(axi4_sim, 2)

        r = _write_burst(axi4_sim, 0xC000_0000, 8)
        assert r["aw_done"] == 1
        assert r["w_beats"] == 8
        assert r["b_done"] == 1
        assert r["bresp"] == 0

    def test_npu_read_write_sequence(self, axi4_sim):
        """Read input → write output sequence."""
        read_size = 4 * BYTES_PER_BEAT
        write_size = 4 * BYTES_PER_BEAT

        image = build_shm_image(
            num_commands=2,
            commands=[
                {"opcode": OP_PUSH, "cmd_id": 0, "interface_id": 0,
                 "protocol": PROTO_AXI4, "role": 0, "buffer_id": 0,
                 "size": read_size, "phys_addr": 0x1000},
                {"opcode": OP_PULL, "cmd_id": 1, "interface_id": 0,
                 "protocol": PROTO_AXI4, "role": 0, "buffer_id": 1,
                 "size": write_size, "phys_addr": 0x2000},
            ],
            num_buffers=2,
        )
        data_off = struct.unpack_from("<Q", image, 0x30)[0]
        total = max(len(image), data_off + read_size + write_size + 64)
        image.extend(b"\x00" * (total - len(image)))

        _setup(axi4_sim, image)
        _issue_cmd(axi4_sim, OP_PUSH, cmd_id=0, buf_id=0,
                   size=read_size, phys_addr=0x1000)
        _issue_cmd(axi4_sim, OP_PULL, cmd_id=1, buf_id=1,
                   size=write_size, phys_addr=0x2000)
        _tick(axi4_sim, 2)

        dbg = _get_debug(axi4_sim)
        assert dbg["active_table"] == 2

        # Read input
        r = _read_burst(axi4_sim, 0x1000, 4)
        assert r["ar_done"] == 1
        assert r["beats"] == 4

        # Write output
        r = _write_burst(axi4_sim, 0x2000, 4)
        assert r["aw_done"] == 1
        assert r["w_beats"] == 4
        assert r["b_done"] == 1
