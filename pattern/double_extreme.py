from .pattern_name import PatternName


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
