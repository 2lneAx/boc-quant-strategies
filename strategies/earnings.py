"""
财报/季报效应策略

逻辑:
  - 财报发布前的"预期"行情：发布日前 N 天买入，赌超预期
  - 财报发布后的"惊喜"效应：超预期则继续持有，低于预期则卖出
  - 银行股财报可预测性强，超预期空间小，适合谨慎使用

说明:
  这是一个基于日历的策略。需要每年更新财报发布日期。
  其实用性依赖于对财报结果的预判能力。

中国银行适用度: ⭐⭐ (银行财报可预期性强，超预期空间有限)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal


class EarningsStrategy(BaseStrategy):
    """
    财报事件驱动。

    参数:
        buy_days_before : 财报发布前 N 日买入 (默认 5)
        hold_days_after : 财报发布后持有 N 日 (默认 10)
        earnings_dates  : 财报发布日期列表
        positive_threshold : 净利润增速阈值，超此值视为"超预期"继续持有 (默认 2%)
    """

    def __init__(self):
        super().__init__(name="财报效应")
        self._params = {
            "buy_days_before": 5,
            "hold_days_after": 10,
            # 中国银行近年财报发布日（年报+一季报+中报+三季报）
            # 格式: (日期, 净利润同比增速%) — 实际数据需每年更新
            "earnings_events": [
                # 年报 (3月底-4月底)
                ("2019-03-29", 4.5), ("2020-04-27", 2.9),
                ("2021-03-30", 12.3), ("2022-03-29", 5.1),
                ("2023-03-30", 2.4), ("2024-03-28", 0.8),
                ("2025-03-28", 1.1),
                # 中报 (8月底)
                ("2019-08-30", 3.7), ("2020-08-28", 1.5),
                ("2021-08-30", 10.6), ("2022-08-30", 6.3),
                ("2023-08-30", 1.7), ("2024-08-29", -1.2),
                ("2025-08-29", 0.0),
            ],
            "positive_threshold": 2.0,
        }

    @property
    def description(self) -> str:
        return (
            f"财报效应 (买入提前{self._params['buy_days_before']}天, "
            f"阈值>{self._params['positive_threshold']}%)"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        buy_before = self._params["buy_days_before"]
        hold_after = self._params["hold_days_after"]
        events = self._params["earnings_events"]
        pos_threshold = self._params["positive_threshold"]

        signals = pd.Series(Signal.HOLD, index=df.index, dtype=int)

        for report_date_str, growth_pct in events:
            report_date = pd.Timestamp(report_date_str)

            # 找财报日对应的 bar
            idx_arr = df.index.get_indexer([report_date], method="bfill")
            report_idx = idx_arr[0]
            if report_idx < 0 or report_idx >= len(df):
                continue

            # 买入: 财报前 buy_before 天
            buy_idx = max(0, report_idx - buy_before)

            # 判断是否超预期
            if growth_pct >= pos_threshold:
                # 超预期 → 持有更久
                sell_idx = min(len(df) - 1, report_idx + hold_after)
            else:
                # 低于预期 → 财报发布次日即卖出
                sell_idx = min(len(df) - 1, report_idx + 1)

            if buy_idx < sell_idx:
                signals.iloc[buy_idx] = Signal.BUY
                signals.iloc[sell_idx] = Signal.SELL

        return self._deduplicate_signals(signals)

    @staticmethod
    def _deduplicate_signals(signals: pd.Series) -> pd.Series:
        cleaned = signals.copy()
        position = 0
        for i in range(len(cleaned)):
            sig = cleaned.iloc[i]
            if sig == Signal.BUY:
                if position == 1:
                    cleaned.iloc[i] = Signal.HOLD
                else:
                    position = 1
            elif sig == Signal.SELL:
                if position == 0:
                    cleaned.iloc[i] = Signal.HOLD
                else:
                    position = 0
        return cleaned
