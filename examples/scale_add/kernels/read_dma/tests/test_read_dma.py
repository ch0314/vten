from vten.cli.scenario import TestScenario


class TestReadDMA(TestScenario):
    kernel = "read_dma"

    # Two configs run back-to-back in ONE sim session. The "replay" config
    # re-runs the same N=1024 transfer after "default" has already latched
    # S_DONE — this exercises the DMA core's re-arm path (fresh start pulse
    # re-latches src addr/length and clears counters, done deasserts). Without
    # the re-arm fix the sticky done=1 makes config 2 falsely insta-pass or the
    # poll time out. A fresh DDR buffer is allocated per config, so the second
    # run also verifies the source address is re-latched.
    configs = [
        {"name": "default"},  # N=1024 (project default)
        {"name": "replay"},   # same N — re-run in-session to exercise re-arm
    ]
