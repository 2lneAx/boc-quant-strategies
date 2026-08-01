"""
上证50 / 沪深300 联动策略

逻辑:
  - 中国银行是上证50成分股，与指数高度相关
  - 监控上证50趋势：指数在 MA(60) 上方且北向资金净流入 → BUY 中行
  - 指数跌破关键均线或资金转向 → SELL
  - 宏观择时 + 个股执行

说明:
  实际运行需要获取指数数据。此处实现用中行自身价格
  模拟宏观趋势信号（取长周期均线代表大势方向）。

中国银行适用度: ⭐⭐⭐⭐ (宏观信号+个股执行，逻辑清晰)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class IndexLinkage(BaseStrategy):
    """
    指数联动宏观择时。

    参数:
        trend_period   : 趋势判断均线周期 (默认 60，代表季线)
        filter_period  : 短期过滤器，避免刚跌破均线就买入 (默认 10)
        trend_strength : 趋势确认天数 (默认 3，连续N天在均线上方才确认)
    """

    def __init__(self):
        super().__init__(name="指数联动")
        self._params = {
            "trend_period": 60,
            "filter_period": 10,
            "trend_strength": 3,
        }

    @property
    def description(self) -> str:
        return (
            f"指数联动 (MA{self._params['trend_period']}趋势确认, "
            f"连续{self._params['trend_strength']}天)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trend_period = self._params["trend_period"]
        filter_period = self._params["filter_period"]
        trend_strength = self._params["trend_strength"]

        close = df["close"]
        volume = df["volume"]

        # ── 宏观趋势信号 ─────────────────────────
        # 长周期均线代表"大势"
        ma_trend = self.ma(close, trend_period)

        # 成交量趋势确认（北向资金替代指标）
        ma_vol = self.ma(volume, trend_period)
        vol_ratio = volume / ma_vol.replace(0, np.nan)

        # 趋势状态: 价格 > 均线 + 成交量放大
        trend_up = (close > ma_trend) & (vol_ratio > 0.8)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)
        position = 0

        warmup = trend_period + trend_strength

        for i in range(warmup, len(df)):
            if pd.isna(ma_trend.iloc[i]):
                continue

            if position == 0:
                # 确认趋势成立：连续 N 天满足趋势条件
                if i >= trend_strength:
                    recent_up = trend_up.iloc[i - trend_strength + 1:i + 1]
                    if recent_up.all():
                        # 额外确认：短期价格 > 短期均线（避免追在最高点）
                        ma_short = close.iloc[max(0, i - filter_period):i + 1].mean()
                        if close.iloc[i] > ma_short * 0.98:
                            signals.iloc[i] = Signal.BUY
                            position = 1
            else:
                # 趋势结束信号：价格跌破长周期均线
                if close.iloc[i] < ma_trend.iloc[i] * 0.98:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 或：连续缩量 + 价格走弱
                elif (vol_ratio.iloc[i] < 0.6 and
                      close.iloc[i] < ma_trend.iloc[i] * 1.02):
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals
