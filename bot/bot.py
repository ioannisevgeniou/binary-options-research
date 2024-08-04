from pattern.pattern_finder import PatternFinder
from features.extractor import Extractor


# Class of the bot for binary options trading
class Bot:
    """
    Initialization Function of the bot.
    It is utilized to initialize the instance variables for the candlestick data, pattern finder, logging, and Double Bottom / Double Top models.
    """

    def __init__(self, candles, logging, model_db, model_dt) -> None:
        self.candles = candles  # Candlestick data
        self.pattern_finder = PatternFinder()  # Object to find trading patterns
        self.logging = logging  # Flag for logging activities
        self.model_db = model_db  # Model for Double Bottom pattern
        self.model_dt = model_dt  # Model for Double Top pattern

    """
    Process candlestick data one by one to check for trading patterns and execute trades.
    _process_candle:
       Args:
          new_candle: A dictionary to hold data for a new candle.
          extractor : Extractor class for extracting features.
       Returns:
          Tuple (signal, close_price, position) where:
             - It returns the signal 1 if a pattern is detected; otherwise, 0.
             - close_price: Closing price of the entry candlestick.
             - position: The trading position (long or short).
    """

    def _process_candle(self, new_candle, extractor):
        # Use the pattern finder to detect patterns in the given candlestick data
        pattern = self.pattern_finder.find_patterns(
            self.candles,
            new_candle,
            extractor,
            self.logging,
            self.model_db,
            self.model_dt,
        )

        # If a pattern is detected, determine the trading position
        if pattern:
            if pattern.pattern_name.value == "Double Bottom":
                position = "long"
            else:
                position = "short"

            # Log the trade details if logging is enabled
            if self.logging:
                print(pattern)
                print(
                    f'Opened {position} position on {new_candle["close"]} at {new_candle["open_time"]}'
                )

            return (
                1,
                new_candle["close"],
                position,
            )  # Return a signal, the close price, and position

        else:
            return 0, None, None  # Return no signal if no pattern is detected

    """
    Iterate dataset to make trades and extract features.
    _pattern_features_service:
        Args:
            time: Binary option expiration time.
        Does forward testing or does data extraction within the specified time interval.
        Logs the trade details, open and close prices, results, number of trades, and win rate if logging is enabled.
        Saves extracted features if the logging is disabled.

    """

    def _pattern_features_service(self, time):
        # Log the start of the process based on logging flag
        if self.logging:
            print(f"Running forward testing \n")
        else:
            print(f"Running data extraction... \n", flush=True)

        in_position = 0  # Indicator for open position
        timespan = 0  # Counter for time in position
        wins = 0  # Counter for successful trades
        total_trades = 0  # Total number of trades
        result = ""  # Result of the trade
        i = -1  # Index for iterating through candles
        extractor = Extractor()  # Initialize feature extractor
        results = []  # List to store trade results

        # Iterate over all candlesticks in the dataset
        while i < len(self.candles) - 1:
            i += 1

            # Check for a trading signal if not already in a position
            if not in_position:
                if i + time < len(self.candles):
                    in_position, entry, position = self._process_candle(
                        self.candles.iloc[i], extractor
                    )
                timespan = 0

            # Manage open position and evaluate its outcome after the specified time
            if in_position:
                if timespan != time:
                    timespan += 1
                else:
                    in_position = 0

                    # Evaluate trade result based on position type
                    if position == "long":
                        if entry < self.candles.iloc[i]["close"]:
                            wins += 1
                            result = "Trade won"
                        else:
                            result = "Trade lost"
                    else:
                        if entry > self.candles.iloc[i]["close"]:
                            wins += 1
                            result = "Trade won"
                        else:
                            result = "Trade lost"
                    total_trades += 1

                    # Log trade details if logging is enabled
                    if self.logging:
                        print(
                            f'Closed position on {self.candles.iloc[i]["close"]} at {self.candles.iloc[i]["open_time"]} \n'
                        )
                        print(result)
                        print(f"Total trades: {total_trades}")
                        print(
                            f"Win rate: {round(wins / total_trades * 100, 2)}% \n",
                            flush=True,
                        )
                    results.append(1 if result == "Trade won" else 0)

                    # Move to the next set of candles after closing a trade
                    self.candles = self.candles[i + 1 :].reset_index(drop=True)
                    i = -1

        # If not logging, save the extracted features
        if not self.logging:
            extractor._save_features(results)
