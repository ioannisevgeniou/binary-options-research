from pattern.pattern_finder import PatternFinder


class Bot:
    def __init__(self, candles, logging) -> None:
        self.candles = candles
        self.pattern_finder = PatternFinder()
        self.logging = logging

    def _process_candle(self, new_candle):
        pattern = self.pattern_finder.find_patterns(self.candles, new_candle)
        if pattern:
            if pattern.pattern_name.value == "Double Bottom":
                position = "long"
            else:
                position = "short"

            if self.logging:
                print(pattern)
                print(
                    f'Opened {position} position on {new_candle["close"]} at {new_candle["open_time"]}'
                )

            return 1, new_candle["close"], position

        else:
            return 0, None, None

    def _pattern_features_service(self):
        if self.logging:
            print(f"Running forward testing \n")
        else:
            print(f"Running data extraction \n")

        in_position = 0
        timespan = 0
        wins = 0
        total_trades = 0
        result = ""
        i = -1

        while i < len(self.candles) - 1:
            i += 1
            if not in_position:
                in_position, entry, position = self._process_candle(
                    self.candles.iloc[i]
                )
                timespan = 0
            if in_position:
                if timespan != 10:
                    timespan += 1
                else:
                    in_position = 0

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

                    self.candles = self.candles[i + 1 :].reset_index(drop=True)
                    i = -1
