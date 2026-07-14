from vten.cli.scenario import TestScenario


class TestInt16X1(TestScenario):
    """16-bit bus: single int16 per beat."""

    kernel = "int16_x1"
