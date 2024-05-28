# Function to check if candle is green
def is_green(candle) -> bool:
    return candle.close > candle.open


# Function to check if candle is red
def is_red(candle) -> bool:
    """Returns True iff `candle` is red"""
    return not is_green(candle)


# Function to count green supertrends
def long_supertrends_count(candle):
    long_superts_count = 0

    if candle["in_uptrend_12_3"]:
        long_superts_count += 1

    if candle["in_uptrend_11_2"]:
        long_superts_count += 1

    if candle["in_uptrend_10_1"]:
        long_superts_count += 1

    return long_superts_count


# Function to count red supertrends
def short_supertrends_count(candle):
    short_superts_count = 0

    if not candle["in_uptrend_12_3"]:
        short_superts_count += 1

    if not candle["in_uptrend_11_2"]:
        short_superts_count += 1

    if not candle["in_uptrend_10_1"]:
        short_superts_count += 1

    return short_superts_count
