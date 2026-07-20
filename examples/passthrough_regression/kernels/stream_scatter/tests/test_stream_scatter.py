from vten.cli.scenario import TestScenario


class TestStreamScatter(TestScenario):
    kernel = "stream_scatter"

    # Two configs run back-to-back in ONE sim session. The "replay" config
    # re-runs the same N=1024 transfer after "default" has already latched
    # done — this exercises the scatter FSM's re-arm path (S_DONE returns to
    # S_IDLE; the next start pulse re-latches both dst addrs/length and clears
    # counters and done). Fresh HBM buffers are allocated per config, so the
    # second run also verifies the destination addresses are re-latched.
    configs = [
        {"name": "default"},  # N=1024 (project default)
        {"name": "replay"},   # same N — re-run in-session to exercise re-arm
    ]
