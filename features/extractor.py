import pandas as pd
from dataset.candle_utils import long_supertrends_count, short_supertrends_count


class Extractor:

    def __init__(self) -> None:
        self.observations = []

    def _save_features(self, results):
        df = pd.DataFrame(self.observations)
        df["result"] = results

        df.to_csv("features/features.csv", index=False)

    def _extract_features(
        self, pattern_name, start, reversal1, msb, reversal2, end, extract
    ):
        observation = {
            "pattern": pattern_name,
            "ema_cross": (
                end["short_term_ema"] > end["medium_term_ema"]
                if pattern_name == "Double Bottom"
                else end["short_term_ema"] < end["medium_term_ema"]
            ),
            "rsi_cross": (
                end["rsi"] > 50 if pattern_name == "Double Bottom" else end["rsi"] < 50
            ),
            "rsi_divergence": (
                reversal2["rsi"] > reversal1["rsi"]
                if pattern_name == "Double Bottom"
                else reversal2["rsi"] < reversal1["rsi"]
            ),
            "rsi_reversal1_deep": (
                reversal1["rsi"] > 70
                if pattern_name == "Double Bottom"
                else reversal1["rsi"] < 30
            ),
            "rsi_reversal2_deep": (
                reversal2["rsi"] > 70
                if pattern_name == "Double Bottom"
                else reversal2["rsi"] < 30
            ),
            "price_divergence": (
                reversal2["close"] < reversal1["close"]
                if pattern_name == "Double Bottom"
                else reversal2["close"] > reversal1["close"]
            ),
            "rsi_reversal1": reversal1["rsi"],
            "rsi_reversal2": reversal2["rsi"],
            "rsi_start": start["rsi"],
            "rsi_msb": msb["rsi"],
            "rsi_end": end["rsi"],
            "atr_reversal1": reversal1["atr"],
            "atr_reversal2": reversal2["atr"],
            "atr_start": start["atr"],
            "atr_msb": msb["atr"],
            "atr_end": end["atr"],
            "cci_reversal1": reversal1["cci"],
            "cci_reversal2": reversal2["cci"],
            "cci_start": start["cci"],
            "cci_msb": msb["cci"],
            "cci_end": end["cci"],
            "cmf_reversal1": reversal1["cmf"],
            "cmf_reversal2": reversal2["cmf"],
            "cmf_start": start["cmf"],
            "cmf_msb": msb["cmf"],
            "cmf_end": end["cmf"],
            "adx_reversal1": reversal1["adx"],
            "adx_reversal2": reversal2["adx"],
            "adx_start": start["adx"],
            "adx_msb": msb["adx"],
            "adx_end": end["adx"],
            "psar_reversal1": (1 if reversal1["psar"] < reversal1["open"] else 0),
            "psar_reversal2": (1 if reversal2["psar"] < reversal2["open"] else 0),
            "psar_start": 1 if start["psar"] < start["open"] else 0,
            "psar_msb": 1 if msb["psar"] < msb["open"] else 0,
            "psar_end": 1 if end["psar"] < end["open"] else 0,
            "short_term_ema_reversal1": (
                1 if reversal1["short_term_ema"] < reversal1["close"] else 0
            ),
            "short_term_ema_reversal2": (
                1 if reversal2["short_term_ema"] < reversal2["close"] else 0
            ),
            "short_term_ema_start": (
                1 if start["short_term_ema"] < start["close"] else 0
            ),
            "short_term_ema_msb": 1 if msb["short_term_ema"] < msb["close"] else 0,
            "short_term_ema_end": 1 if end["short_term_ema"] < end["close"] else 0,
            "medium_term_ema_reversal1": (
                1 if reversal1["medium_term_ema"] < reversal1["close"] else 0
            ),
            "medium_term_ema_reversal2": (
                1 if reversal2["medium_term_ema"] < reversal2["close"] else 0
            ),
            "medium_term_ema_start": (
                1 if start["medium_term_ema"] < start["close"] else 0
            ),
            "medium_term_ema_msb": 1 if msb["medium_term_ema"] < msb["close"] else 0,
            "medium_term_ema_end": 1 if end["medium_term_ema"] < end["close"] else 0,
            "long_super_reversal1": long_supertrends_count(reversal1),
            "long_super_reversal2": long_supertrends_count(reversal2),
            "long_super_start": long_supertrends_count(start),
            "long_super_msb": long_supertrends_count(msb),
            "long_super_end": long_supertrends_count(end),
            "short_super_reversal1": short_supertrends_count(reversal1),
            "short_super_reversal2": short_supertrends_count(reversal2),
            "short_super_start": short_supertrends_count(start),
            "short_super_msb": short_supertrends_count(msb),
            "short_super_end": short_supertrends_count(end),
        }

        if extract:
            self.observations.append(observation)
        else:
            return observation
