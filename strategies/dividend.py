"""
股息分红策略

逻辑:
  - 在股权登记日前 N 天买入，获取股息
  - 除权除息后择机卖出（等待填权）
  - 核心假设：高股息股票具有填权倾向

说明:
  这是一个基于日历的策略。中国银行通常每年 6-7 月分红。
  实际使用时需要每年更新分红日期。
  默认参数包含了中行近年的分红登记日期。

中国银行适用度: ⭐⭐⭐ (股息率 5-7%，适合中长期持有+波段增强)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class DividendStrategy(BaseStrategy):
    """
    股息分红日历策略。

    参数:
        buy_days_before : 登记日前 N 个交易日买入 (默认 10)
        sell_days_after : 除权后 N 个交易日卖出 (默认 20)
        dividend_dates  : 股权登记日列表 (需每年更新)
    """

    def __init__(self):
        super().__init__(name="股息分红策略")
        self._params = {
            "buy_days_before": 10,
            "sell_days_after": 20,
            # 中国银行近年股权登记日（实际日期，需每年维护）
            "dividend_dates": [
                "2016-06-23", "2017-07-13", "2018-07-12",
                "2019-07-10", "2020-07-14", "2021-07-09",
                "2022-07-14", "2023-07-13", "2024-07-11",
                "2025-07-10",
            ],
        }

    @property
    def description(self) -> str:
        return (
            f"股息分红策略 (买入提前{self._params['buy_days_before']}天, "
            f"持有{self._params['sell_days_after']}天)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        buy_before = self._params["buy_days_before"]
        sell_after = self._params["sell_days_after"]
        div_dates_str = self._params["dividend_dates"]

        # 转换分红日期
        div_dates = pd.to_datetime(div_dates_str)

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        # 找出每个登记日对应的交易日
        for div_date in div_dates:
            # 找登记日当天或之后最近的交易日
            mask_on_or_after = df.index >= div_date
            if not mask_on_or_after.any():
                continue

            # 登记日对应的 bar 索引
            div_idx = df.index.get_indexer([div_date], method="bfill")[0]
            if div_idx < 0 or div_idx >= len(df):
                continue

            # 买入日: 登记日往前 buy_days_before 个交易日
            buy_idx = max(0, div_idx - buy_before)
            # 卖出日: 登记日往后 sell_days_after 个交易日
            sell_idx = min(len(df) - 1, div_idx + sell_after)

            if buy_idx < sell_idx:
                signals.iloc[buy_idx] = Signal.BUY
                signals.iloc[sell_idx] = Signal.SELL

        # 处理重叠信号：如果连续买入日之间有冲突，后面的买入不覆盖前面的持仓
        return self._deduplicate_signals(signals)

    @staticmethod
    def _deduplicate_signals(signals: pd.Series) -> pd.Series:
        """
        清理重叠的信号序列。同一时间只有一个买入和一个卖出，
        如果在已有持仓时再出买入信号，忽略之。
        """
        cleaned = signals.copy()
        position = 0
        for i in range(len(cleaned)):
            sig = cleaned.iloc[i]
            if sig == Signal.BUY:
                if position == 1:
                    cleaned.iloc[i] = Signal.HOLD  # 已有持仓，忽略
                else:
                    position = 1
            elif sig == Signal.SELL:
                if position == 0:
                    cleaned.iloc[i] = Signal.HOLD  # 无持仓，忽略
                else:
                    position = 0
        return cleaned
