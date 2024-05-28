import pandas as pd

from .double_bottom_finder import DoubleBottomFinder
from .double_top_finder import DoubleTopFinder


class PatternFinder:

    def __init__(self) -> None:
        self.db_finder = DoubleBottomFinder()
        self.dt_finder = DoubleTopFinder()

    def find_patterns(
        self, candles: pd.DataFrame, new_candle: pd.Series, extractor, logging
    ):
        db = self.db_finder.find_double_bottom(candles, new_candle, extractor, logging)
        if db:
            return db

        dt = self.dt_finder.find_double_top(candles, new_candle, extractor, logging)
        if dt:
            return dt
