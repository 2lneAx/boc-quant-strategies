"""
OBV (能量潮) 背离策略

逻辑:
  - 价格创 N 日新低，但 OBV 未创 N 日新低 → 底背离 BUY
  - 价格创 N 日新高，但 OBV 未创 N 日新高 → 顶背离 SELL
  - 背离后需要价格确认反转才入场

中国银行适用度: ⭐⭐⭐
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class OBVDivergence(BaseStrategy):
    """
    OBV 背离。

    参数:
        lookback     : 检测高/低点的回溯窗口 (默认 20)
        confirm_bars : 背离后需要 N 根 K 线确认反转才入场 (默认 2)
        use_macd     : 是否用 MACD 辅助确认 (默认 True)
    """

    def __init__(self):
        super().__init__(name="OBV背离")
        self._params = {
            "lookback": 20,
            "confirm_bars": 2,
            "use_macd": True,
        }

    @property
    def description(self) -> str:
        return (
            f"OBV背离 (lookback={self._params['lookback']}, "
            f"confirm={self._params['confirm_bars']}bars)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = self._params["lookback"]
        confirm_bars = self._params["confirm_bars"]
        use_macd = self._params["use_macd"]

        close = df["close"]
        volume = df["volume"]

        # ── OBV 计算 ──────────────────────────────
        obv = pd.Series(0.0, index=df.index)
        obv_val = 0.0
        for i in range(len(df)):
            if i == 0:
                obv.iloc[i] = 0
                continue
            if close.iloc[i] > close.iloc[i - 1]:
                obv_val += volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv_val -= volume.iloc[i]
            obv.iloc[i] = obv_val

        # ── MACD 辅助 ─────────────────────────────
        macd_line = pd.Series(np.nan, index=df.index)
        signal_line = pd.Series(np.nan, index=df.index)
        if use_macd:
            macd_line, signal_line = self._calc_macd(close)

        # ── 滚动极值 ──────────────────────────────
        price_high = close.rolling(window=lookback).max()
        price_low = close.rolling(window=lookback).min()
        obv_high = obv.rolling(window=lookback).max()
        obv_low = obv.rolling(window=lookback).min()

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)
        position = 0

        warmup = lookback + 26 + confirm_bars

        for i in range(warmup, len(df)):
            if position == 0:
                # 底背离：价格新低 + OBV 未新低
                price_new_low = close.iloc[i] <= price_low.iloc[i] * 1.01
                obv_not_new_low = obv.iloc[i] > obv_low.iloc[i] * 1.01

                if price_new_low and obv_not_new_low:
                    # 确认: 随后 confirm_bars 天内价格回升
                    if self._confirm_reversal(close, i, confirm_bars, direction="up"):
                        if use_macd:
                            if macd_line.iloc[i] > signal_line.iloc[i]:
                                signals.iloc[i] = Signal.BUY
                                position = 1
                        else:
                            signals.iloc[i] = Signal.BUY
                            position = 1

            else:
                # 顶背离：价格新高 + OBV 未新高
                price_new_high = close.iloc[i] >= price_high.iloc[i] * 0.99
                obv_not_new_high = obv.iloc[i] < obv_high.iloc[i] * 0.99

                if price_new_high and obv_not_new_high:
                    if self._confirm_reversal(close, i, confirm_bars, direction="down"):
                        signals.iloc[i] = Signal.SELL
                        position = 0
                # 固定止损 8%
                elif close.iloc[i] < close.iloc[max(0, i - 20):i + 1].max() * 0.92:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals

    @staticmethod
    def _confirm_reversal(series: pd.Series, idx: int, bars: int,
                          direction: str) -> bool:
        """确认价格在 bars 天内反转"""
        end = min(idx + bars, len(series) - 1)
        if end <= idx:
            return False
        future_prices = series.iloc[idx:end + 1]
        trigger_price = series.iloc[idx]
        if direction == "up":
            return any(future_prices > trigger_price * 1.01)
        else:
            return any(future_prices < trigger_price * 0.99)

    @staticmethod
    def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9):
        """计算 MACD 线和信号线"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
