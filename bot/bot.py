import pandas as pd

from pattern.pattern_finder import PatternFinder


class Bot:
    def __init__(self, candles) -> None:
        self.candles = candles
        self.pattern_finder = PatternFinder()

    def _process_candle(self, new_candle):
        pattern = self.pattern_finder.find_patterns(self.candles, new_candle)
        if pattern:
            if pattern.pattern_name.value == "Double Bottom":
                position = "long"
            else:
                position = "short"

            print(
                f'Opened {position} position on {new_candle["close"]} at {new_candle["open_time"]}'
            )

            return 1, new_candle["close"], position

        else:
            return 0, None, None
