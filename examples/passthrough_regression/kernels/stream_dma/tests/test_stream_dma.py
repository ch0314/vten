from vten.cli.scenario import TestScenario


class TestStreamDma(TestScenario):
    kernel = "stream_dma"

    # Two configs run back-to-back in ONE sim session. The "replay" config
    # re-runs the same N=1024 transfer after "default" has already latched
    # done — this exercises the DMA engine's re-arm path (DMA_DONE returns to
    # DMA_IDLE; the next start pulse re-latches dst addr/length and clears
    # done). A fresh DDR buffer is allocated per config, so the second run
    # also verifies the destination address is re-latched.
    configs = [
        {"name": "default"},  # N=1024 (project default)
        {"name": "replay"},   # same N — re-run in-session to exercise re-arm
    ]
