"""
银行股配对交易策略

逻辑:
  - 利用中行与工行/建行/农行的高度相关性
  - 计算价差标准化 Z-Score
  - Z > +2: 中行相对高估 → 卖出中行 (做多另一只)
  - Z < -2: 中行相对低估 → 买入中行
  - A 股限制: 不实际做空，只做多低估端

注意:
  本策略仅使用中行自身数据，用滚动窗口内的相对强弱
  来模拟"相对于银行板块"的强弱——作为配对交易的近似实现。
  真正的配对交易需要同行业股票数据协同计算。

中国银行适用度: ⭐⭐ (A股做空限制多，纯多头配对只能单向做)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class PairsTrading(BaseStrategy):
    """
    银行股配对交易（简化版）。

    用中行相对于自身历史波动区间的 Z-Score 做均值回归 ——
    等价于"中行配对自身均值"的交易逻辑。

    参数:
        lookback    : 价差计算窗口 (默认 60)
        entry_z     : 入场 Z-Score 阈值 (默认 2.0)
        exit_z      : 出场 Z-Score 阈值 (默认 0.0，回归均值即出)
        peer_weight : 同行比较权重 (0=纯自身, 1=最大同行影响, 默认 0.5)
    """

    def __init__(self):
        super().__init__(name="配对交易(简化)")
        self._params = {
            "lookback": 60,
            "entry_z": 2.0,
            "exit_z": 0.0,
            "peer_weight": 0.5,
        }

    @property
    def description(self) -> str:
        return (
            f"配对交易简化版 (lookback={self._params['lookback']}, "
            f"entry_z=±{self._params['entry_z']})"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = self._params["lookback"]
        entry_z = self._params["entry_z"]
        exit_z = self._params["exit_z"]

        close = df["close"]

        # 计算对数收益率
        log_ret = np.log(close / close.shift(1))

        # 滚动 Z-Score：当前值偏离滚动均值的标准化距离
        rolling_mean = close.rolling(window=lookback).mean()
        rolling_std = close.rolling(window=lookback).std()
        z_score = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        position = 0   # 0=空仓, 1=做多中行

        for i in range(lookback + 1, len(df)):
            z = z_score.iloc[i]
            if pd.isna(z):
                continue

            if position == 0:
                # 中行相对低估 → 买入
                if z < -entry_z:
                    signals.iloc[i] = Signal.BUY
                    position = 1
            else:
                # 回归均值 → 平仓
                if z > exit_z:
                    signals.iloc[i] = Signal.SELL
                    position = 0
                # 极端反向 (被高估) → 也平仓
                elif z > entry_z:
                    signals.iloc[i] = Signal.SELL
                    position = 0

        return signals
