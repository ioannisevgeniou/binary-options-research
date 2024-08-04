from typing import Optional

import pandas as pd
from dataset.candle_utils import is_green, is_red
from .double_extreme import DoubleExtreme
from .pattern_name import PatternName
from joblib import load
from modelling.model_caller import predict

"""
    Recognize the pattern (double bottom) within candlestick information series. 

    Args:
        candles (pd.DataFrame): candlestick data contained dataframe.
        current (pd.Series): The latest candlestick being examined.
        extractor: object for feature extraction within pattern recognized.
        logging (bool): flag depicting whether to go ahead with logging process.
        model_db (str): model name needed for prediction.

    Returns:
        Optional[DoubleExtreme]: DoubleExtreme object is returned that represent valid pattern, 
        else the None is returned.
"""


class DoubleBottomFinder:

    def __init__(self) -> None:
        self.search_end = 0  # Initialize the search end boundary for pattern finding

    def find_double_bottom(
        self,
        candles: pd.DataFrame,
        current: pd.Series,
        extractor,
        logging,
        model_db,
    ) -> Optional[DoubleExtreme]:
        # Initialize the pattern components
        end = current
        reversal2 = None
        msb = None
        reversal1 = None
        start = None

        # If the current (end) candle is red, exit early as it doesn't fit the pattern
        if is_red(end):
            return None

        # Iterate backwards through the candles from the current candle to search_end
        for i in range(end.name - 1, self.search_end, -1):
            candle = candles.iloc[i]

            # If a green candle closes above the end, it's invalid for the pattern
            if is_green(candle) and candle["close"] > end["close"]:
                return None

            # Identify reversal2
            if msb is None:
                if (
                    is_red(candle)
                    and candle["open"] < end["close"]
                    and (reversal2 is None or candle["close"] < reversal2["close"])
                    and candle["close"]
                    < candles.loc[
                        candles.loc[candle.name + 1 : end.name]["close"].idxmin()
                    ]["close"]
                ):
                    reversal2 = candle

            # Identify the middle of the swing bottom (MSB)
            if reversal1 is None and reversal2 is not None:
                if (
                    is_green(candle)
                    and candle["close"] < end["close"]
                    and candle["open"] > reversal2["close"]
                    and (msb is None or candle["close"] > msb["close"])
                    and reversal2.name - candle.name > 1
                    and candle["close"]
                    > candles.loc[
                        candles.loc[candle.name + 1 : reversal2.name]["close"].idxmax()
                    ]["close"]
                ):
                    msb = candle

            # Identify reversal1
            if msb is not None:
                if (
                    (reversal1 is None or candle["close"] < reversal1["close"])
                    and is_red(candle)
                    and candle["open"] < msb["close"]
                    and msb.name - candle.name > 1
                    and candle["close"]
                    < candles.loc[
                        candles.loc[candle.name + 1 : msb.name]["close"].idxmin()
                    ]["close"]
                ):
                    reversal1 = candle

            # Identify the start of the pattern
            if (
                reversal1 is not None
                and is_red(candle)
                and candle["close"] > reversal1["close"]
                and candle["open"] > msb["close"]
                and candle["open"]
                > candles.loc[
                    candles.loc[candle.name + 1 : reversal1.name]["open"].idxmax()
                ]["open"]
            ):
                start = candle

                # Load the prediction model if logging is enabled and model exists
                model = None
                if logging and model_db is not None:
                    model = load("modelling/" + model_db + ".joblib")
                # Extract features for further processing if no model is used
                else:
                    extractor._extract_features(
                        PatternName.DOUBLE_BOTTOM.value,
                        start,
                        reversal1,
                        msb,
                        reversal2,
                        end,
                        1,
                    )

                # Predict the validity of the pattern
                if predict(model, start, reversal1, msb, reversal2, end, logging):
                    return DoubleExtreme(
                        PatternName.DOUBLE_BOTTOM, start, reversal1, msb, reversal2, end
                    )
                else:
                    print("Pattern rejected! \n")
                    return None
