import pandas as pd

from .double_bottom_finder import DoubleBottomFinder
from .double_top_finder import DoubleTopFinder

"""
    This class looks to figure out double bottom & double top patterns within information about candlestick
    and uses class DoubleBottomFinder & DoubleTopFinder.

    Args:
        candles (pd.DataFrame): candlestick data contained dataframe.
        new_candle (pd.Series): newer candlestick being examined.
        extractor: object for feature extraction within pattern recognized.
        logging (bool): flag depicting whether to go ahead with logging process.
        model_db (str): model name needed for double bottom prediction.
        model_dt (str): model name needed for double top prediction.

    Returns:
        Optional[DoubleExtreme]: DoubleExtreme object is returned that represent valid pattern, 
        else the None is returned.
"""


class PatternFinder:
    # Initialize pattern finders for double bottom and double top
    def __init__(self) -> None:
        self.db_finder = DoubleBottomFinder()  # Double bottom pattern finder
        self.dt_finder = DoubleTopFinder()  # Double top pattern finder

    def find_patterns(
        self,
        candles: pd.DataFrame,
        new_candle: pd.Series,
        extractor,
        logging,
        model_db,
        model_dt,
    ):
        # Attempt to find a double bottom pattern
        db = self.db_finder.find_double_bottom(
            candles, new_candle, extractor, logging, model_db
        )
        if db:
            return db

        # Attempt to find a double top pattern
        dt = self.dt_finder.find_double_top(
            candles, new_candle, extractor, logging, model_dt
        )
        if dt:
            return dt
