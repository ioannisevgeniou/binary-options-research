def is_green(candle) -> bool:
    """Returns True iff `candle` is green"""
    return candle.close > candle.open


def is_red(candle) -> bool:
    """Returns True iff `candle` is red"""
    return not is_green(candle)
