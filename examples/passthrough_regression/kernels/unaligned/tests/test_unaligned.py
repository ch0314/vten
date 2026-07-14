from vten.cli.scenario import TestScenario


class TestUnaligned(TestScenario):
    """256-bit bus with N=100: tensor size not aligned to 32-byte beat boundary."""

    kernel = "unaligned"
