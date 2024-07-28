from .pattern_name import PatternName

"""
    Double extreme pattern represented, like the double bottom or the double top type, within data of candlestick.

    Args:
        pattern_name (PatternName): identified pattern's name.
        start: candlestick wherein patterns commence.
        reversal1: 1st reversal spot within pattern.
        msb: middle important spot in between consecutive reversals.
        reversal2: 2nd reversal spot within pattern.
        end: candlestick wherein patterns ends.

    Returns:
        None
"""


class DoubleExtreme:

    def __init__(
        self, pattern_name: PatternName, start, reversal1, msb, reversal2, end
    ) -> None:
        self.pattern_name = pattern_name
        self.start = start
        self.reversal1 = reversal1
        self.msb = msb
        self.reversal2 = reversal2
        self.end = end

    def __str__(self) -> str:
        return (
            f"{self.pattern_name.value}\n"
            f'Start - {self.start["open_time"]}\n'
            f'Reversal1 - {self.reversal1["open_time"]}\n'
            f'MSB - {self.msb["open_time"]}\n'
            f'Reversal2 - {self.reversal2["open_time"]}\n'
            f'End - {self.end["open_time"]}\n'
        )
