from enum import Enum


class PatternName(Enum):
    """
    Enumerate candlestick patterns.

    Attributes:
        DOUBLE_BOTTOM (str): depict double bottom pattern.
        DOUBLE_TOP (str): depict double top pattern.
    """

    DOUBLE_BOTTOM = "Double Bottom"
    DOUBLE_TOP = "Double Top"
