"""
布林带均值回归策略

逻辑:
  - 价格触及下轨 (mean - k*std) 超卖 → BUY
  - 价格回归中轨 → 平仓 (SELL)
  - 价格触及上轨 → 可选做空信号 (SELL)
  - 风控: 跌破下轨额外 margin 止损

中国银行适用度: ⭐⭐⭐⭐ (震荡时间长，均值回归命中率高)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class BollingerBreakout(BaseStrategy):
    """
    布林带均值回归。

    参数:
        period  : 均线窗口 (默认 20)
        k       : 标准差倍数 (默认 2.0)
        stop_loss_margin : 跌破下轨后额外下跌幅度止损 (默认 3%)
    """

    def __init__(self):
        super().__init__(name="布林带均值回归")
        self._params = {
            "period": 20,
            "k": 2.0,
            "stop_loss_margin": 0.03,
        }

    @property
    def description(self) -> str:
        return (
            f"布林带均值回归 (period={self._params['period']}, "
            f"k={self._params['k']}, sl={self._params['stop_loss_margin']:.0%})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period = self._params["period"]
        k = self._params["k"]
        sl_margin = self._params["stop_loss_margin"]

        close = df["close"]

        # 布林带计算
        middle = self.ma(close, period)
        sigma = self.std(close, period)
        upper = middle + k * sigma
        lower = middle - k * sigma

        # 带宽百分比 (标准化)
        bandwidth_pct = (close - lower) / (upper - lower)  # 0=下轨, 1=上轨

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        position = 0       # 0=空仓, 1=持仓
        entry_price = 0.0  # 入场价

        for i in range(period, len(df)):
            if pd.isna(lower.iloc[i]):
                continue

            c = close.iloc[i]
            lo = lower.iloc[i]
            mi = middle.iloc[i]

            if position == 0:
                # 触及下轨超卖 → 买入
                if c <= lo * (1 + 0.005):  # 允许 0.5% 容差
                    signals.iloc[i] = Signal.BUY
                    position = 1
                    entry_price = c
            else:
                # 回归中轨 → 卖出
                if c >= mi:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 止损: 跌破下轨后继续下跌
                elif c < entry_price * (1 - sl_margin):
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals
