from vten.cli.scenario import TestScenario


class TestMmLoopback(TestScenario):
    kernel = "mm_loopback"

    # Two configs run back-to-back in ONE sim session. The "replay" config
    # re-runs the same N=1024 loopback after "default" has already latched
    # done — this exercises the re-arm path (S_DONE returns to S_IDLE and the
    # next ctrl pulse gives a fresh start edge; done deasserts). Without
    # `pulse: true` on ctrl, the latched start level yields no second edge, so
    # config 2 would see stale done=1 and its PULL would starve. Fresh DDR
    # buffers per config also verify the src/dst addresses are re-read.
    configs = [
        {"name": "default"},  # N=1024 (project default)
        {"name": "replay"},   # same N — re-run in-session to exercise re-arm
    ]
