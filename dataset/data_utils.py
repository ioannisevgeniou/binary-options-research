import numpy as np
import pandas as pd
import pandas_ta as ta

from dataset import indicators_tracker


# Function to prepare dataset
def initialiaze_dataset(candles):
    format_times(candles)
    candles = format_dataset(candles)
    candles = calculate_indicators(candles)
    return candles


# Function to format times
def format_times(candles):
    candles[["open_time", "close_time"]] = candles[["open_time", "close_time"]].apply(
        lambda col: pd.to_datetime(pd.to_numeric(col), unit="ms")
    )


# Function to format dataset
def format_dataset(candles):
    candles.open = candles.open.astype(float)
    candles.high = candles.high.astype(float)
    candles.low = candles.low.astype(float)
    candles.close = candles.close.astype(float)
    candles.volume = candles.volume.astype(float)

    if indicators_tracker.EMA_LENGTHS["SHORT_TERM"]:
        candles["short_term_ema"] = None
    if indicators_tracker.EMA_LENGTHS["MEDIUM_TERM"]:
        candles["medium_term_ema"] = None
    if indicators_tracker.EMA_LENGTHS["LONG_TERM"]:
        candles["long_term_ema"] = None

    if indicators_tracker.RSI_LENGTH:
        candles["rsi"] = None

    if indicators_tracker.ATR_LENGTH:
        candles["atr"] = None

    candles["tr"] = None
    candles["atr_12"] = None
    candles["atr_11"] = None
    candles["atr_10"] = None
    candles["is_local_maximum"] = None
    candles["is_local_minimum"] = None

    candles["in_uptrend_12_3"] = None
    candles["upperband_12_3"] = None
    candles["lowerband_12_3"] = None
    candles["supert_12_3"] = None

    candles["in_uptrend_11_2"] = None
    candles["upperband_11_2"] = None
    candles["lowerband_11_2"] = None
    candles["supert_11_2"] = None

    candles["in_uptrend_10_1"] = None
    candles["upperband_10_1"] = None
    candles["lowerband_10_1"] = None
    candles["supert_10_1"] = None

    if indicators_tracker.CMF_LENGTH:
        candles["cmf"] = None
    if indicators_tracker.CCI_LENGTH:
        candles["cci"] = None
    if indicators_tracker.ADX_LENGTH:
        candles["adx"] = None

    if indicators_tracker.MA_LENGTHS["SHORT_TERM"]:
        candles["short_term_ma"] = None
    if indicators_tracker.MA_LENGTHS["MEDIUM_TERM"]:
        candles["medium_term_ma"] = None
    if indicators_tracker.MA_LENGTHS["LONG_TERM"]:
        candles["long_term_ma"] = None

    if indicators_tracker.STOCHRSI["LENGTH"]:
        candles["stochrsi_k"] = None
        candles["stochrsi_d"] = None

    if indicators_tracker.MACD["FAST"]:
        candles["macd"] = None
        candles["macd_histogram"] = None
        candles["macd_signal"] = None

    if indicators_tracker.BBANDS["LENGTH"]:
        candles["bbands_lower"] = None
        candles["bbands_mid"] = None
        candles["bbands_upper"] = None
        candles["bbands_bandwidth"] = None
        candles["bbands_percent"] = None

    if indicators_tracker.ICHIMOKU["TENKAN"]:
        candles["ichimoku_span_a"] = None
        candles["ichimoku_span_b"] = None
        candles["ichimoku_tenkan"] = None
        candles["ichimoku_kijun"] = None

    if indicators_tracker.PSAR["AF0"]:
        candles["psar"] = None

    return candles


# Function to add ema column in dataset
def create_ema_column(candles, name, length):
    if length:
        candles[name] = ta.ema(candles.close, length=length).round(2)


# Function to add ma column in dataset
def create_ma_column(candles, name, length):
    if length:
        candles[name] = ta.sma(candles.close, length=length, talib=True).round(2)


# Function to calculate indicators
def calculate_indicators(candles):
    create_ema_column(
        candles, "short_term_ema", indicators_tracker.EMA_LENGTHS["SHORT_TERM"]
    )
    create_ema_column(
        candles, "medium_term_ema", indicators_tracker.EMA_LENGTHS["MEDIUM_TERM"]
    )
    create_ema_column(
        candles, "long_term_ema", indicators_tracker.EMA_LENGTHS["LONG_TERM"]
    )

    if indicators_tracker.RSI_LENGTH:
        candles.rsi = ta.rsi(candles.close, length=indicators_tracker.RSI_LENGTH).round(
            2
        )

    if indicators_tracker.ATR_LENGTH:
        candles["atr"] = ta.atr(
            candles.high,
            candles.low,
            candles.close,
            length=indicators_tracker.ATR_LENGTH,
        ).round(2)

    candles_copy = candles.copy(deep=True)

    candles["tr"] = tr(candles_copy)
    candles["atr_12"] = atr(candles_copy, 12)
    candles["atr_11"] = atr(candles_copy, 11)
    candles["atr_10"] = atr(candles_copy, 10)

    maxima = local_maxima(candles_copy)
    candles["is_local_maximum"] = maxima[0]
    candles["is_local_minimum"] = maxima[1]

    supertrends(candles)

    if indicators_tracker.CMF_LENGTH:
        candles["cmf"] = ta.cmf(
            candles.high,
            candles.low,
            candles.close,
            candles.volume,
            length=indicators_tracker.CMF_LENGTH,
        ).round(2)

    if indicators_tracker.CCI_LENGTH:
        candles["cci"] = ta.cci(
            candles.high,
            candles.low,
            candles.close,
            length=indicators_tracker.CCI_LENGTH,
        ).round(2)

    if indicators_tracker.ADX_LENGTH:
        candles["adx"] = ta.adx(
            candles.high,
            candles.low,
            candles.close,
            length=indicators_tracker.ADX_LENGTH,
        )[f"ADX_{indicators_tracker.ADX_LENGTH}"].round(2)

    create_ma_column(
        candles, "short_term_ma", indicators_tracker.MA_LENGTHS["SHORT_TERM"]
    )
    create_ma_column(
        candles, "medium_term_ma", indicators_tracker.MA_LENGTHS["MEDIUM_TERM"]
    )
    create_ma_column(
        candles, "long_term_ma", indicators_tracker.MA_LENGTHS["LONG_TERM"]
    )

    if indicators_tracker.STOCHRSI["LENGTH"]:
        stochrsi = ta.stochrsi(
            candles.close,
            length=indicators_tracker.STOCHRSI["LENGTH"],
            rsi_length=indicators_tracker.STOCHRSI["RSI_LENGTH"],
            k=indicators_tracker.STOCHRSI["K"],
            d=indicators_tracker.STOCHRSI["D"],
        )
        candles["stochrsi_k"] = stochrsi[
            f"STOCHRSIk_{indicators_tracker.STOCHRSI['LENGTH']}_{indicators_tracker.STOCHRSI['RSI_LENGTH']}_{indicators_tracker.STOCHRSI['K']}_{indicators_tracker.STOCHRSI['D']}"
        ].round(2)
        candles["stochrsi_d"] = stochrsi[
            f"STOCHRSId_{indicators_tracker.STOCHRSI['LENGTH']}_{indicators_tracker.STOCHRSI['RSI_LENGTH']}_{indicators_tracker.STOCHRSI['K']}_{indicators_tracker.STOCHRSI['D']}"
        ].round(2)

    if indicators_tracker.MACD["FAST"]:
        macd = ta.macd(
            candles.close,
            fast=indicators_tracker.MACD["FAST"],
            slow=indicators_tracker.MACD["SLOW"],
            signal=indicators_tracker.MACD["SIGNAL"],
        )
        candles["macd"] = macd[
            f"MACD_{indicators_tracker.MACD['FAST']}_{indicators_tracker.MACD['SLOW']}_{indicators_tracker.MACD['SIGNAL']}"
        ].round(2)
        candles["macd_histogram"] = macd[
            f"MACDh_{indicators_tracker.MACD['FAST']}_{indicators_tracker.MACD['SLOW']}_{indicators_tracker.MACD['SIGNAL']}"
        ].round(2)
        candles["macd_signal"] = macd[
            f"MACDs_{indicators_tracker.MACD['FAST']}_{indicators_tracker.MACD['SLOW']}_{indicators_tracker.MACD['SIGNAL']}"
        ].round(2)

    if indicators_tracker.BBANDS["LENGTH"]:
        bbands = ta.bbands(candles.close, length=indicators_tracker.BBANDS["LENGTH"])
        candles["bbands_lower"] = bbands[
            f"BBL_{indicators_tracker.BBANDS['LENGTH']}_{indicators_tracker.BBANDS['STD']}"
        ].round(2)
        candles["bbands_mid"] = bbands[
            f"BBM_{indicators_tracker.BBANDS['LENGTH']}_{indicators_tracker.BBANDS['STD']}"
        ].round(2)
        candles["bbands_upper"] = bbands[
            f"BBU_{indicators_tracker.BBANDS['LENGTH']}_{indicators_tracker.BBANDS['STD']}"
        ].round(2)
        candles["bbands_bandwidth"] = (
            bbands[
                f"BBB_{indicators_tracker.BBANDS['LENGTH']}_{indicators_tracker.BBANDS['STD']}"
            ]
            .div(100)
            .round(4)
        )
        candles["bbands_percent"] = bbands[
            f"BBP_{indicators_tracker.BBANDS['LENGTH']}_{indicators_tracker.BBANDS['STD']}"
        ].round(2)

    if indicators_tracker.ICHIMOKU["TENKAN"]:
        ichimoku = ta.ichimoku(
            candles.high,
            candles.low,
            candles.close,
            tenkan=indicators_tracker.ICHIMOKU["TENKAN"],
            kijun=indicators_tracker.ICHIMOKU["KIJUN"],
            senkou=indicators_tracker.ICHIMOKU["SENKOU"],
        )

        candles["ichimoku_span_a"] = ichimoku[0][
            f"ISA_{indicators_tracker.ICHIMOKU['TENKAN']}"
        ].round(2)
        candles["ichimoku_span_b"] = ichimoku[0][
            f"ISB_{indicators_tracker.ICHIMOKU['KIJUN']}"
        ].round(2)
        candles["ichimoku_tenkan"] = ichimoku[0][
            f"ITS_{indicators_tracker.ICHIMOKU['TENKAN']}"
        ].round(2)
        candles["ichimoku_kijun"] = ichimoku[0][
            f"IKS_{indicators_tracker.ICHIMOKU['KIJUN']}"
        ].round(2)

    if indicators_tracker.PSAR["AF0"]:
        psar = ta.psar(
            candles.high,
            candles.low,
            candles.close,
            af0=indicators_tracker.PSAR["AF0"],
            af=indicators_tracker.PSAR["AF"],
            max_af=indicators_tracker.PSAR["MAX_AF"],
        )

        psar_long = psar[
            f"PSARl_{indicators_tracker.PSAR['AF0']}_{indicators_tracker.PSAR['MAX_AF']}"
        ].fillna(0)
        psar_short = psar[
            f"PSARs_{indicators_tracker.PSAR['AF0']}_{indicators_tracker.PSAR['MAX_AF']}"
        ].fillna(0)
        candles["psar"] = (psar_long + psar_short).round(2)
        candles.loc[:, "psar"] = candles["psar"].replace(0, np.nan)

    return candles


# Auxilliary function to atr indicator
def tr(data):
    data["previous_close"] = data["close"].shift(1)
    data["high-low"] = abs(data["high"] - data["low"])
    data["high-pc"] = abs(data["high"] - data["previous_close"])
    data["low-pc"] = abs(data["low"] - data["previous_close"])

    tr = data[["high-low", "high-pc", "low-pc"]].max(axis=1).round(2)

    return tr


# Function for local maximum
def local_maxima(data):
    data["next_close"] = data["close"].shift(-1)
    data["next_open"] = data["open"].shift(-1)

    local_maximum = np.where(
        (data["close"] > data["open"]) & (data["next_open"] > data["next_close"]),
        True,
        False,
    )

    local_minimum = np.where(
        (data["close"] < data["open"]) & (data["next_open"] < data["next_close"]),
        True,
        False,
    )

    return [local_maximum, local_minimum]


# Function to calculate atr
def atr(data, period):
    data["tr"] = tr(data)
    atr = data["tr"].rolling(period).mean()
    return atr


# Function to calculate supertrends
def supertrend(df, period=7, atr_multiplier=3):
    df = df.copy()
    hl2 = (df["high"] + df["low"]) / 2
    df["atr"] = atr(df, period)
    df["upperband"] = hl2 + (atr_multiplier * df["atr"])
    df["lowerband"] = hl2 - (atr_multiplier * df["atr"])
    df["in_uptrend"] = True

    for index, _ in df.iloc[1:].iterrows():
        current = df.loc[index].name
        previous = current - 1

        if df["close"].loc[current] > df["upperband"].loc[previous]:
            df["in_uptrend"][current] = True
        elif df["close"].loc[current] < df["lowerband"].loc[previous]:
            df["in_uptrend"].loc[current] = False
        else:
            df["in_uptrend"].loc[current] = df["in_uptrend"].loc[previous]

            if (
                df["in_uptrend"].loc[current]
                and df["lowerband"].loc[current] < df["lowerband"].loc[previous]
            ):
                df["lowerband"].loc[current] = df["lowerband"].loc[previous]

            if (
                not df["in_uptrend"].loc[current]
                and df["upperband"].loc[current] > df["upperband"].loc[previous]
            ):
                df["upperband"].loc[current] = df["upperband"].loc[previous]

    upperband = f"upperband_{period}_{atr_multiplier}"
    lowerband = f"lowerband_{period}_{atr_multiplier}"
    supert = f"supert_{period}_{atr_multiplier}"
    in_uptrend = f"in_uptrend_{period}_{atr_multiplier}"

    df[supert] = df.apply(
        lambda row: row.lowerband if row.in_uptrend else row.upperband, axis=1
    )
    df.rename(
        columns={
            "upperband": upperband,
            "lowerband": lowerband,
            "in_uptrend": in_uptrend,
        },
        inplace=True,
    )
    return df[[in_uptrend, upperband, lowerband, supert]]


# Function to calculate supertrends
def supertrends(df):
    hl2 = (df["high"] + df["low"]) / 2

    atr_periods = [12, 11, 10]
    multipliers = [3, 2, 1]

    for period, multiplier in zip(atr_periods, multipliers):
        atr_period = f"atr_{period}"

        df[f"in_uptrend_{period}_{multiplier}"] = True
        df[f"upperband_{period}_{multiplier}"] = (
            hl2 + (multiplier * df[atr_period])
        ).round(2)
        df[f"lowerband_{period}_{multiplier}"] = (
            hl2 - (multiplier * df[atr_period])
        ).round(2)
        df[f"supert_{period}_{multiplier}"] = None

    for index, _ in df.iloc[1:].iterrows():
        current = df.loc[index].name
        previous = current - 1

        for period, multiplier in zip(atr_periods, multipliers):
            upperband = f"upperband_{period}_{multiplier}"
            lowerband = f"lowerband_{period}_{multiplier}"
            supert = f"supert_{period}_{multiplier}"
            in_uptrend = f"in_uptrend_{period}_{multiplier}"

            if df.at[current, "close"] > df.at[previous, upperband]:
                df.at[current, in_uptrend] = True
            elif df.at[current, "close"] < df.at[previous, lowerband]:
                df.at[current, in_uptrend] = False
            else:
                df.at[current, in_uptrend] = df.at[previous, in_uptrend]

            if (
                df.at[current, lowerband] < df.at[previous, lowerband]
                and df.at[previous, "close"] > df.at[previous, lowerband]
            ):
                df.at[current, lowerband] = df.at[previous, lowerband]

            if (
                df.at[current, upperband] > df.at[previous, upperband]
                and df.at[previous, "close"] < df.at[previous, upperband]
            ):
                df.at[current, upperband] = df.at[previous, upperband]

            df.at[current, supert] = (
                df.at[current, lowerband].round(2)
                if df.at[current, in_uptrend]
                else df.at[current, upperband].round(2)
            )

    return df


# Function to calculate supertrends
def supertrends_for_new_candle(df):
    hl2 = (df["high"] + df["low"]) / 2

    atr_periods = [12, 11, 10]
    multipliers = [3, 2, 1]

    for period, multiplier in zip(atr_periods, multipliers):
        atr_period = f"atr_{period}"

        df[f"upperband_{period}_{multiplier}"] = hl2 + (multiplier * df[atr_period])
        df[f"lowerband_{period}_{multiplier}"] = hl2 - (multiplier * df[atr_period])
        df[f"in_uptrend_{period}_{multiplier}"] = True

    for index, row in df.iloc[1:].iterrows():
        current = df.loc[index].name
        previous = current - 1

        for period, multiplier in zip(atr_periods, multipliers):
            upperband = f"upperband_{period}_{multiplier}"
            lowerband = f"lowerband_{period}_{multiplier}"
            supert = f"supert_{period}_{multiplier}"
            in_uptrend = f"in_uptrend_{period}_{multiplier}"

            if df["close"].loc[current] > df[upperband].loc[previous]:
                df[in_uptrend][current] = True
            elif df["close"].loc[current] < df[lowerband].loc[previous]:
                df[in_uptrend].loc[current] = False
            else:
                df[in_uptrend].loc[current] = df[in_uptrend].loc[previous]

                if (
                    df[in_uptrend].loc[current]
                    and df[lowerband].loc[current] < df[lowerband].loc[previous]
                ):
                    df[lowerband].loc[current] = df[lowerband].loc[previous]

                if (
                    not df[in_uptrend].loc[current]
                    and df[upperband].loc[current] > df[upperband].loc[previous]
                ):
                    df[upperband].loc[current] = df[upperband].loc[previous]

    for period, multiplier in zip(atr_periods, multipliers):
        df[f"supert_{period}_{multiplier}"] = df.apply(
            lambda row: (
                row[f"lowerband_{period}_{multiplier}"]
                if row[f"in_uptrend_{period}_{multiplier}"]
                else row[f"upperband_{period}_{multiplier}"]
            ),
            axis=1,
        )

    return df
