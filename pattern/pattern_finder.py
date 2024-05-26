import pandas as pd

from .double_bottom_finder import DoubleBottomFinder
from .double_top_finder import DoubleTopFinder


class PatternFinder:

    def __init__(self) -> None:
        self.db_finder = DoubleBottomFinder()
        self.dt_finder = DoubleTopFinder()

    def find_patterns(self, candles: pd.DataFrame, new_candle: pd.Series):
        db = self.db_finder.find_double_bottom(candles, new_candle)
        if db:
            return db

        dt = self.dt_finder.find_double_top(candles, new_candle)
        if dt:
            return dt
