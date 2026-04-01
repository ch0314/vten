from vten.cli.run import TestScenario


class TestOffset(TestScenario):
    """Offset kernel standalone: add offset_value=1 to each byte."""

    kernel = "offset"
