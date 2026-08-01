"""
放量突破策略

逻辑:
  - 价格突破 N 日均线 + 成交量放大至均量的 M 倍 → BUY
  - 量能萎缩 + 价格跌破支撑 → SELL
  - 量价配合过滤假突破，比纯价格突破更可靠

中国银行适用度: ⭐⭐⭐⭐ (大市值，量价信号含金量高)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class VolumeBreakout(BaseStrategy):
    """
    放量突破。

    参数:
        price_period   : 价格均线周期 (默认 20)
        vol_period     : 成交量均线周期 (默认 20)
        vol_multiplier : 放量倍数，当日量 > 均量 * multiplier 视为放量 (默认 1.5)
        exit_vol_shrink: 缩量倍数，当日量 < 均量 * multiplier 视为缩量 (默认 0.5)
        hold_days_min  : 最短持有天数，避免频繁交易 (默认 3)
    """

    def __init__(self):
        super().__init__(name="放量突破")
        self._params = {
            "price_period": 10,     # 优化: 20→10, 更灵敏的突破判断
            "vol_period": 30,       # 优化: 20→30, 更平滑的量能基准
            "vol_multiplier": 1.2,  # 优化: 1.5→1.2, 较低门槛捕捉早期信号
            "exit_vol_shrink": 0.3, # 优化: 0.5→0.3, 更耐心持有
            "hold_days_min": 1,     # 优化: 3→1, 不阻止止损
        }

    @property
    def description(self) -> str:
        return (
            f"放量突破 (priceMA={self._params['price_period']}, "
            f"vol×{self._params['vol_multiplier']})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        price_period = self._params["price_period"]
        vol_period = self._params["vol_period"]
        vol_mult = self._params["vol_multiplier"]
        exit_shrink = self._params["exit_vol_shrink"]
        hold_min = self._params["hold_days_min"]

        close = df["close"]
        volume = df["volume"]

        # 均线
        ma_price = self.ma(close, price_period)
        ma_vol = self.ma(volume, vol_period)

        # 量比
        vol_ratio = volume / ma_vol.replace(0, np.nan)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        position = 0
        entry_price = 0.0
        entry_bar = -999

        warmup = max(price_period, vol_period) + 1

        for i in range(warmup, len(df)):
            if pd.isna(ma_price.iloc[i]) or pd.isna(vol_ratio.iloc[i]):
                continue

            c = close.iloc[i]
            vr = vol_ratio.iloc[i]
            mp = ma_price.iloc[i]

            if position == 0:
                # 突破均线 + 放量 → 买入
                if c > mp and vr > vol_mult:
                    signals.iloc[i] = Signal.BUY
                    position = 1
                    entry_price = c
                    entry_bar = i
            else:
                bars_held = i - entry_bar

                # 必须持有最小天数
                if bars_held < hold_min:
                    continue

                # 缩量 + 跌破均线 → 卖出
                if c < mp and vr < exit_shrink:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 止损 5%
                elif c < entry_price * 0.95:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 止盈 10%
                elif c > entry_price * 1.10:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals
