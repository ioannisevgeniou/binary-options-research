from typing import Optional

import pandas as pd
from dataset.candle_utils import is_green, is_red

from .double_extreme import DoubleExtreme
from .pattern_name import PatternName


class DoubleTopFinder:

    def __init__(self) -> None:
        self.search_end = 0

    def find_double_top(
        self, candles: pd.DataFrame, current: pd.Series
    ) -> Optional[DoubleExtreme]:
        end = current
        reversal2 = None
        msb = None
        reversal1 = None
        start = None

        if is_green(end):
            return None

        for i in range(end.name - 1, self.search_end, -1):
            candle = candles.iloc[i]
            if msb is None:
                if (
                    (reversal2 is None or candle["close"] > reversal2["close"])
                    and is_green(candle)
                    and candle["open"] > end["open"]
                ):
                    reversal2 = candle

            if reversal1 is None and reversal2 is not None:
                if (
                    is_red(candle)
                    and candle["close"] > end["close"]
                    and candle["open"] < reversal2["open"]
                    and (msb is None or candle["close"] < msb["close"])
                ):
                    msb = candle

            if msb is not None:
                if (
                    (reversal1 is None or candle["close"] > reversal1["close"])
                    and is_green(candle)
                    and candle["open"] > reversal2["open"]
                ):
                    reversal1 = candle

            if (
                reversal1 is not None
                and is_green(candle)
                and candle["open"] < reversal2["open"]
                and candle["open"] < msb["close"]
            ):
                start = candle
                self.search_end = reversal2.name

                return DoubleExtreme(
                    PatternName.DOUBLE_TOP, start, reversal1, msb, reversal2, end
                )
