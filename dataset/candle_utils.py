"""
Check whether if the candle has green color (closing price > opening price).

is_green:
    Args:
        candle: An object having the attributes viz; 'close' and 'open'.
    Returns:
        bool: To flag True if candle has green color, otherwise False.    
"""


def is_green(candle) -> bool:
    return candle.close > candle.open


"""
Check whether if the candle has red color (closing price < opening price).

is_red:
    Args:
        candle: An object having the attributes viz; 'close' and 'open'.
    Returns:
        bool: To flag True if candle has red color, otherwise False.    
"""


def is_red(candle) -> bool:
    """Returns True iff `candle` is red"""
    return not is_green(candle)


"""
Count the number of green supertrends in a candle.

long_supertrends_count:
    Args:
        candle: A dictionary containing supertrend indicators with keys 'in_uptrend_12_3', 
                'in_uptrend_11_2', and 'in_uptrend_10_1'.
    Returns:
        int: The number of green supertrends.
"""


def long_supertrends_count(candle):
    long_superts_count = 0

    if candle["in_uptrend_12_3"]:
        long_superts_count += 1

    if candle["in_uptrend_11_2"]:
        long_superts_count += 1

    if candle["in_uptrend_10_1"]:
        long_superts_count += 1

    return long_superts_count


"""
Counting red supertrends number in a candle.

short_supertrends_count:
    Args:
        candle: A dictionary to contain the supertrend indicators having the keys viz; 'in_uptrend_12_3', 
                'in_uptrend_11_2', and 'in_uptrend_10_1'.
    Returns:
        int: The red supertrends count.
"""


def short_supertrends_count(candle):
    short_superts_count = 0

    if not candle["in_uptrend_12_3"]:
        short_superts_count += 1

    if not candle["in_uptrend_11_2"]:
        short_superts_count += 1

    if not candle["in_uptrend_10_1"]:
        short_superts_count += 1

    return short_superts_count
