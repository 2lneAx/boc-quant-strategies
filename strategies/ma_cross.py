"""
双均线交叉策略

逻辑:
  - 快线上穿慢线 (金叉) → BUY
  - 快线下穿慢线 (死叉) → SELL
  - ADX 过滤器: ADX < 20 时不交易，过滤震荡市假信号

中国银行适用度: ⭐⭐⭐ (需配合趋势过滤，否则震荡市磨损严重)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class MACross(BaseStrategy):
    """
    双均线交叉趋势跟踪。

    参数:
        fast    : 快线周期 (默认 5)
        slow    : 慢线周期 (默认 20)
        use_adx : 是否使用 ADX 过滤震荡市 (默认 True)
        adx_period : ADX 周期 (默认 14)
        adx_threshold : ADX 阈值，低于此值不交易 (默认 20)
    """

    def __init__(self):
        super().__init__(name="双均线交叉")
        self._params = {
            "fast": 5,
            "slow": 20,
            "use_adx": True,
            "adx_period": 14,
            "adx_threshold": 20,
        }

    @property
    def description(self) -> str:
        return (
            f"双均线交叉 MA({self._params['fast']},{self._params['slow']})"
            + (" +ADX过滤" if self._params["use_adx"] else "")
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = self._params["fast"]
        slow = self._params["slow"]
        use_adx = self._params["use_adx"]
        adx_period = self._params["adx_period"]
        adx_threshold = self._params["adx_threshold"]

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ma_fast = self.ema(close, fast)
        ma_slow = self.ema(close, slow)

        # ADX 计算
        adx = pd.Series(np.nan, index=df.index)
        if use_adx:
            adx = self._calc_adx(high, low, close, adx_period)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)
        position = 0

        warmup = max(slow, adx_period + 1, fast)

        for i in range(warmup, len(df)):
            f = ma_fast.iloc[i]
            s = ma_slow.iloc[i]
            f_prev = ma_fast.iloc[i - 1]
            s_prev = ma_slow.iloc[i - 1]

            if pd.isna(f) or pd.isna(s) or pd.isna(f_prev) or pd.isna(s_prev):
                continue

            # ADX 过滤
            if use_adx and not pd.isna(adx.iloc[i]):
                if adx.iloc[i] < adx_threshold:
                    continue  # 震荡市，不产生信号

            if position == 0:
                # 金叉（快线从下方上穿慢线）
                if f_prev <= s_prev and f > s:
                    signals.iloc[i] = Signal.BUY
                    position = 1
            else:
                # 死叉（快线从上方下穿慢线）
                if f_prev >= s_prev and f < s:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals

    # ── ADX 计算 (Welles Wilder) ──────────────────

    @staticmethod
    def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        """计算 ADX — 平均趋向指数"""
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()

        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up > down) & (up > 0)] = up
        minus_dm[(down > up) & (down > 0)] = down

        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx
