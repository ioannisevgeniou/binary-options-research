import pandas as pd
import pandas_ta as ta

from talipp.indicators import ATR, RSI
from talipp.ohlcv import OHLCVFactory
from dataset.candle_utils import is_green

# Set as None to not generate indicator
EMA_LENGTHS = {
    # Multitimeframe(mtf) EMA
    # (higher timeframe) * (ema value) / (lower timeframe)
    "SHORT_TERM": 50,
    "MEDIUM_TERM": 200,
    "LONG_TERM": 800,
}
RSI_LENGTH = 14
ATR_LENGTH = 14
CMF_LENGTH = 20
CCI_LENGTH = 30
ADX_LENGTH = 14
MA_LENGTHS = {"SHORT_TERM": 50, "MEDIUM_TERM": 200, "LONG_TERM": 800}
STOCHRSI = {"LENGTH": 14, "RSI_LENGTH": 14, "K": 3, "D": 3}  # Set only this to None
MACD = {"FAST": 12, "SLOW": 26, "SIGNAL": 9}  # Set only this to None
BBANDS = {"LENGTH": 20, "STD": 2.0}  # Set only this to None
ICHIMOKU = {"TENKAN": 9, "KIJUN": 26, "SENKOU": 52}  # Set only this to None
PSAR = {"AF0": 0.02, "AF": 0.02, "MAX_AF": 0.2}  # Set only this to None


class IndicatorsTracker:
    def __init__(self, candles) -> None:
        self.rsis = RSI(RSI_LENGTH, candles.close.tolist())
        ohlcv = OHLCVFactory.from_matrix2(
            [
                candles.open.tolist(),
                candles.high.tolist(),
                candles.low.tolist(),
                candles.close.tolist(),
                [],
                candles.open_time.tolist(),
            ]
        )
        self.atrs_12 = ATR(12, ohlcv)
        self.atrs_11 = ATR(11, ohlcv)
        self.atrs_10 = ATR(10, ohlcv)

    # TODO: Add test
    def calculate_indicators_for_new_candle(self, candles, new_candle, last_candle):
        # WHY CUT THE DATAFRAME???
        # Give less candles to improve speed???
        # BUT ...
        # For example: ATR 14 needs 14 candles(=length) to be calculated
        # Giving only the last 14 candles results in wrong value
        # Giving min the last 100 candles results in right value

        # EMA 800 needs min 3500 last candles for right value
        last_3500 = candles.iloc[-3500:]
        new_candle.short_term_ema = self.calculate_ema_for_new_candle(
            last_3500, EMA_LENGTHS["SHORT_TERM"]
        )
        new_candle.medium_term_ema = self.calculate_ema_for_new_candle(
            last_3500, EMA_LENGTHS["MEDIUM_TERM"]
        )
        new_candle.long_term_ema = self.calculate_ema_for_new_candle(
            last_3500, EMA_LENGTHS["LONG_TERM"]
        )

        new_candle.rsi = self.calculate_rsi_for_new_candle(new_candle)

        # ATR needs min 100 last candles for right value
        last_100 = candles.iloc[-100:]
        new_candle["atr"] = (
            ta.atr(last_100.high, last_100.low, last_100.close, length=ATR_LENGTH)
            .round(2)
            .iloc[-1]
        )

        new_candle["tr"] = self.tr_for_candle(new_candle, last_candle)

        new_candle["atr_12"] = self.atr_for_new_candle(new_candle, candles, 12)
        new_candle["atr_11"] = self.atr_for_new_candle(new_candle, candles, 11)
        new_candle["atr_10"] = self.atr_for_new_candle(new_candle, candles, 10)

        new_candle["is_local_maximum"] = False
        new_candle["is_local_minimum"] = False

        self.supertrends_for_new_candle(new_candle, last_candle)

        # CMF needs candles=length for right values
        last_20 = candles.iloc[-20:]
        new_candle["cmf"] = (
            ta.cmf(
                last_20.high,
                last_20.low,
                last_20.close,
                last_20.volume,
                length=CMF_LENGTH,
            )
            .round(2)
            .iloc[-1]
        )

        # CCI needs candles=length for right values
        last_30 = candles.iloc[-30:]
        new_candle["cci"] = (
            ta.cci(last_30.high, last_30.low, last_30.close, length=CCI_LENGTH)
            .round(2)
            .iloc[-1]
        )

        # ADX needs min 500 last candles for right value
        last_500 = candles.iloc[-500:]
        new_candle["adx"] = (
            ta.adx(last_500.high, last_500.low, last_500.close, length=ADX_LENGTH)[
                f"ADX_{ADX_LENGTH}"
            ]
            .round(2)
            .iloc[-1]
        )

        # MA needs candles=length for right value
        last_1000 = candles.iloc[-1000:]
        new_candle.short_term_ma = self.calculate_ma_for_new_candle(
            last_1000, MA_LENGTHS["SHORT_TERM"]
        )
        new_candle.medium_term_ma = self.calculate_ma_for_new_candle(
            last_1000, MA_LENGTHS["MEDIUM_TERM"]
        )
        new_candle.long_term_ma = self.calculate_ma_for_new_candle(
            last_1000, MA_LENGTHS["LONG_TERM"]
        )

        # STOCHRSI needs min 100 last candles for right value
        stochrsi = ta.stochrsi(
            last_100.close,
            length=STOCHRSI["LENGTH"],
            rsi_length=STOCHRSI["RSI_LENGTH"],
            k=STOCHRSI["K"],
            d=STOCHRSI["D"],
        )
        new_candle["stochrsi_k"] = (
            stochrsi[
                f"STOCHRSIk_{STOCHRSI['LENGTH']}_{STOCHRSI['RSI_LENGTH']}_{STOCHRSI['K']}_{STOCHRSI['D']}"
            ]
            .round(2)
            .iloc[-1]
        )
        new_candle["stochrsi_d"] = (
            stochrsi[
                f"STOCHRSId_{STOCHRSI['LENGTH']}_{STOCHRSI['RSI_LENGTH']}_{STOCHRSI['K']}_{STOCHRSI['D']}"
            ]
            .round(2)
            .iloc[-1]
        )

        # MACD needs min 100 last candles for right value
        macd = ta.macd(
            last_100.close, fast=MACD["FAST"], slow=MACD["SLOW"], signal=MACD["SIGNAL"]
        )
        new_candle["macd"] = (
            macd[f"MACD_{MACD['FAST']}_{MACD['SLOW']}_{MACD['SIGNAL']}"]
            .round(2)
            .iloc[-1]
        )
        new_candle["macd_histogram"] = (
            macd[f"MACDh_{MACD['FAST']}_{MACD['SLOW']}_{MACD['SIGNAL']}"]
            .round(2)
            .iloc[-1]
        )
        new_candle["macd_signal"] = (
            macd[f"MACDs_{MACD['FAST']}_{MACD['SLOW']}_{MACD['SIGNAL']}"]
            .round(2)
            .iloc[-1]
        )

        # BBANDS needs min 1000 last candles for right value
        bbands = ta.bbands(last_1000.close, length=BBANDS["LENGTH"])
        new_candle["bbands_lower"] = (
            bbands[f"BBL_{BBANDS['LENGTH']}_{BBANDS['STD']}"].round(2).iloc[-1]
        )
        new_candle["bbands_mid"] = (
            bbands[f"BBM_{BBANDS['LENGTH']}_{BBANDS['STD']}"].round(2).iloc[-1]
        )
        new_candle["bbands_upper"] = (
            bbands[f"BBU_{BBANDS['LENGTH']}_{BBANDS['STD']}"].round(2).iloc[-1]
        )
        new_candle["bbands_bandwidth"] = (
            bbands[f"BBB_{BBANDS['LENGTH']}_{BBANDS['STD']}"].div(100).round(4).iloc[-1]
        )
        new_candle["bbands_percent"] = (
            bbands[f"BBP_{BBANDS['LENGTH']}_{BBANDS['STD']}"].round(2).iloc[-1]
        )

        # ICHIMOKU needs min 1000 last candles for right value
        ichimoku = ta.ichimoku(
            last_1000.high,
            last_1000.low,
            last_1000.close,
            tenkan=ICHIMOKU["TENKAN"],
            kijun=ICHIMOKU["KIJUN"],
            senkou=ICHIMOKU["SENKOU"],
        )
        new_candle["ichimoku_span_a"] = (
            ichimoku[0][f"ISA_{ICHIMOKU['TENKAN']}"].round(2).iloc[-1]
        )
        new_candle["ichimoku_span_b"] = (
            ichimoku[0][f"ISB_{ICHIMOKU['KIJUN']}"].round(2).iloc[-1]
        )
        # conversion line
        new_candle["ichimoku_tenkan"] = (
            ichimoku[0][f"ITS_{ICHIMOKU['TENKAN']}"].round(2).iloc[-1]
        )
        # base line
        new_candle["ichimoku_kijun"] = (
            ichimoku[0][f"IKS_{ICHIMOKU['KIJUN']}"].round(2).iloc[-1]
        )

        # PSAR needs min 4500 last candles for right value
        last_4500 = candles.iloc[-4500:]
        psar = ta.psar(
            last_4500.high,
            last_4500.low,
            last_4500.close,
            af0=PSAR["AF0"],
            af=PSAR["AF"],
            max_af=PSAR["MAX_AF"],
        )
        psar_long = psar[f"PSARl_{PSAR['AF0']}_{PSAR['MAX_AF']}"].fillna(0)
        psar_short = psar[f"PSARs_{PSAR['AF0']}_{PSAR['MAX_AF']}"].fillna(0)
        new_candle["psar"] = (psar_long.iloc[-1] + psar_short.iloc[-1]).round(2)

        return new_candle

    def calculate_local_maxima_for_last_candle(self, candles, last_candle, new_candle):
        candles.loc[last_candle.name, "is_local_maximum"] = not is_green(
            new_candle
        ) and is_green(last_candle)

        candles.loc[last_candle.name, "is_local_minimum"] = is_green(
            new_candle
        ) and not is_green(last_candle)

        return last_candle

    def calculate_ema_for_new_candle(self, candles, period):
        return ta.ema(candles.close, length=period).round(2).iloc[-1]

    def calculate_ma_for_new_candle(self, candles, period):
        return ta.sma(candles.close, length=period).round(2).iloc[-1]

    def calculate_rsi_for_new_candle(self, new_candle):
        self.rsis.add_input_value(new_candle.close)
        return round(self.rsis[-1], 2)

    def calculate_atr_for_new_candle(self, new_candle, period):
        if period == 12:
            self.atrs_12.add_input_value(new_candle)
            return self.atrs_12[-1]
        elif period == 11:
            self.atrs_11.add_input_value(new_candle)
            return self.atrs_11[-1]
        elif period == 10:
            self.atrs_10.add_input_value(new_candle)
            return self.atrs_10[-1]
        else:
            raise ValueError()

    def tr(self, data):
        data["previous_close"] = data["close"].shift(1)
        data["high-low"] = abs(data["high"] - data["low"])
        data["high-pc"] = abs(data["high"] - data["previous_close"])
        data["low-pc"] = abs(data["low"] - data["previous_close"])

        tr = data[["high-low", "high-pc", "low-pc"]].max(axis=1)

        return tr

    def tr_for_candle(self, candle, previous_candle):
        tr = max(
            abs(candle.high - candle.low),
            abs(candle.high - previous_candle.close),
            abs(candle.low - previous_candle.close),
        ).round(2)
        return tr

    def atr_for_new_candle(self, new_candle, data, period):
        tr_sum = (
            data["tr"].loc[new_candle.name - period + 1 : new_candle.name].sum()
            + new_candle["tr"]
        )
        return (tr_sum / period).round(2)

    def atr(self, data, period):
        copy = data.copy(deep=True)
        copy["tr"] = self.tr(copy)
        atr = copy["tr"].rolling(period).mean()

        return atr

    def supertrends_for_new_candle(self, new_candle, last_candle):
        atr_periods = [12, 11, 10]
        multipliers = [3, 2, 1]

        for period, multiplier in zip(atr_periods, multipliers):
            upperband = f"upperband_{period}_{multiplier}"
            lowerband = f"lowerband_{period}_{multiplier}"
            supert = f"supert_{period}_{multiplier}"
            in_uptrend = f"in_uptrend_{period}_{multiplier}"

            hl2 = (new_candle["high"] + new_candle["low"]) / 2
            new_candle[in_uptrend] = True

            new_candle[upperband] = (
                hl2 + (multiplier * new_candle[f"atr_{period}"])
            ).round(2)
            new_candle[lowerband] = (
                hl2 - (multiplier * new_candle[f"atr_{period}"])
            ).round(2)

            if new_candle["close"] > last_candle[upperband]:
                new_candle[in_uptrend] = True
            elif new_candle["close"] < last_candle[lowerband]:
                new_candle[in_uptrend] = False
            else:
                new_candle[in_uptrend] = last_candle[in_uptrend]

            if (
                new_candle[lowerband] < last_candle[lowerband]
                and last_candle["close"] > last_candle[lowerband]
            ):
                new_candle[lowerband] = last_candle[lowerband]

            if (
                new_candle[upperband] > last_candle[upperband]
                and last_candle["close"] < last_candle[upperband]
            ):
                new_candle[upperband] = last_candle[upperband]

            if new_candle[in_uptrend]:
                new_candle[supert] = round(new_candle[lowerband], 2)
            else:
                new_candle[supert] = round(new_candle[upperband], 2)

        return new_candle
