from vten.cli.scenario import TestScenario


class TestInt16X7(TestScenario):
    """112-bit bus: 7 int16 per beat (14 bytes, odd element count)."""

    kernel = "int16_x7"
