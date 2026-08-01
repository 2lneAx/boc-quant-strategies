"""
RSI 超买超卖策略

逻辑:
  - RSI < oversold 阈值 → BUY
  - RSI > overbought 阈值 → SELL
  - 银行股波动小，阈值设为 35/65 比传统 30/70 更合适

中国银行适用度: ⭐⭐⭐
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class RSIReversion(BaseStrategy):
    """
    RSI 超买超卖反转。

    参数:
        period      : RSI 计算周期 (默认 14)
        oversold    : 超卖阈值 (默认 35，银行股窄幅波动)
        overbought  : 超买阈值 (默认 65)
        require_cross : True=需要 RSI 穿过阈值才触发, False=进入区域即触发
    """

    def __init__(self):
        super().__init__(name="RSI超买超卖")
        self._params = {
            "period": 14,
            "oversold": 35,
            "overbought": 65,
            "require_cross": True,
        }

    @property
    def description(self) -> str:
        return (
            f"RSI超买超卖 (period={self._params['period']}, "
            f"os={self._params['oversold']}, ob={self._params['overbought']})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        period = self._params["period"]
        oversold = self._params["oversold"]
        overbought = self._params["overbought"]
        require_cross = self._params["require_cross"]

        close = df["close"]
        rsi = self.rsi(close, period)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)
        position = 0

        for i in range(period + 1, len(df)):
            rsi_now = rsi.iloc[i]
            rsi_prev = rsi.iloc[i - 1]

            if pd.isna(rsi_now) or pd.isna(rsi_prev):
                continue

            if position == 0:
                if require_cross:
                    # RSI 从下方向上穿越 oversold
                    if rsi_prev < oversold and rsi_now >= oversold:
                        signals.iloc[i] = Signal.BUY
                        position = 1
                else:
                    if rsi_now < oversold:
                        signals.iloc[i] = Signal.BUY
                        position = 1
            else:
                if require_cross:
                    # RSI 从上方向下穿越 overbought
                    if rsi_prev > overbought and rsi_now <= overbought:
                        signals.iloc[i] = Signal.SELL
                        position = 0
                else:
                    if rsi_now > overbought:
                        signals.iloc[i] = Signal.SELL
                        position = 0

        return signals
