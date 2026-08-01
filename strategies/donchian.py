"""
唐奇安通道突破策略 (Donchian Channel)

逻辑:
  - 价格突破 N 日最高价 → BUY
  - 价格跌破 N 日最低价 → SELL
  - 经典海龟交易系统的核心之一
  - 配合 ATR 动态止损

中国银行适用度: ⭐⭐ (趋势策略，银行股突破后延续性一般)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class DonchianBreakout(BaseStrategy):
    """
    唐奇安通道突破。

    参数:
        lookback     : 通道回溯周期 (默认 20)
        atr_period   : ATR 周期，用于动态止损 (默认 14)
        atr_stop     : 止损为入场价 - atr_stop * ATR (默认 2.0)
        take_profit  : 止盈为入场价 + take_profit * ATR (默认 3.0)，0 为不止盈
    """

    def __init__(self):
        super().__init__(name="唐奇安通道突破")
        self._params = {
            "lookback": 20,
            "atr_period": 14,
            "atr_stop": 2.0,
            "take_profit": 0.0,
        }

    @property
    def description(self) -> str:
        return (
            f"唐奇安通道突破 (lookback={self._params['lookback']}, "
            f"ATR止损={self._params['atr_stop']}x)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = self._params["lookback"]
        atr_period = self._params["atr_period"]
        atr_stop = self._params["atr_stop"]
        take_profit = self._params["take_profit"]

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # 通道上下轨
        upper = high.rolling(window=lookback).max().shift(1)  # 前 N 日最高价
        lower = low.rolling(window=lookback).min().shift(1)   # 前 N 日最低价

        # ATR
        atr = self._calc_atr(high, low, close, atr_period)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        position = 0
        entry_price = 0.0
        entry_atr = 0.0

        warmup = lookback + atr_period + 1

        for i in range(warmup, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(atr.iloc[i]):
                continue

            c = close.iloc[i]
            h = high.iloc[i]
            l = low.iloc[i]
            a = atr.iloc[i]

            if position == 0:
                # 突破上轨 → 买入
                if h > upper.iloc[i]:
                    signals.iloc[i] = Signal.BUY
                    position = 1
                    entry_price = c
                    entry_atr = a
            else:
                # 跌破下轨 → 卖出
                if l < lower.iloc[i]:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # ATR 动态止损
                elif c <= entry_price - atr_stop * entry_atr:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # ATR 动态止盈（如果启用）
                elif take_profit > 0 and c >= entry_price + take_profit * entry_atr:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals

    @staticmethod
    def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": abs(high - close.shift(1)),
            "lc": abs(low - close.shift(1)),
        }).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
