"""
PB 估值区间择时策略

逻辑:
  - 银行股的核心估值指标是市净率 (PB)
  - 中行历史 PB 在 0.45x - 1.0x 之间波动
  - PB 处于历史低分位 → 低估，逐步加仓
  - PB 处于历史高分位 → 高估，逐步减仓
  - 这是长期配置型策略，非短线交易

说明:
  真实 PB = 股价 / 每股净资产。此处用滚动窗口内的价格分位数
  作为 PB 分位数的近似，因为银行 PB 波动主要由股价驱动
  （每股净资产变化缓慢且可预测）。

中国银行适用度: ⭐⭐⭐⭐⭐ (银行股最核心的估值锚)

参考 PB 区间:
  PB < 0.5 : 极度低估 (历史底部)
  PB 0.5-0.6 : 低估
  PB 0.6-0.8 : 合理
  PB > 0.8 : 高估
  PB > 1.0 : 泡沫
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class PBValuation(BaseStrategy):
    """
    PB 估值区间择时。

    参数:
        lookback     : 滚动窗口（交易日），用于估算历史 PB 分位数 (默认 504，约2年)
        buy_percentile  : 买入分位阈值，价格低于此分位时买入 (默认 20，即价格在历史底部20%)
        sell_percentile : 卖出分位阈值，价格高于此分位时卖出 (默认 80，即价格在历史顶部20%)
        add_position_pct : 价格每下跌此幅度加仓 (默认 5%)
        reduce_position_pct : 价格每上涨此幅度减仓 (默认 5%)
        max_position   : 最大仓位比例 (默认 1.0，即满仓)
    """

    def __init__(self):
        super().__init__(name="PB估值择时")
        self._params = {
            "lookback": 252,           # 优化: 504→252, 1年窗口更灵活
            "buy_percentile": 30,      # 优化: 20→30, 不必等极端低估
            "sell_percentile": 90,     # 优化: 80→90, 让盈利充分奔跑
            "add_position_pct": 0.05,
            "reduce_position_pct": 0.05,
            "max_position": 1.0,
        }

    @property
    def description(self) -> str:
        return (
            f"PB估值择时 (lookback={self._params['lookback']}d, "
            f"买入<P{self._params['buy_percentile']}, 卖出>P{self._params['sell_percentile']})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = self._params["lookback"]
        buy_pct = self._params["buy_percentile"]
        sell_pct = self._params["sell_percentile"]

        close = df["close"]

        # ── 滚动价格分位数（PB 分位近似） ─────
        # 因为银行每股净资产变化缓慢，价格波动是 PB 波动的主因
        rolling_pct = close.rolling(window=lookback, min_periods=lookback // 2).rank(pct=True) * 100

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        position = 0     # 0=空仓, 1=持仓
        entry_price = 0.0

        for i in range(lookback, len(df)):
            rp = rolling_pct.iloc[i]
            if pd.isna(rp):
                continue

            c = close.iloc[i]

            if position == 0:
                # 价格处于历史低分位 → 低估买入
                if rp <= buy_pct:
                    signals.iloc[i] = Signal.BUY
                    position = 1
                    entry_price = c
            else:
                # 价格处于历史高分位 → 高估卖出
                if rp >= sell_pct:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 止损: 买入后继续下跌超 15%（PB 大幅缩水）
                elif c < entry_price * 0.85:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals
