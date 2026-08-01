"""
基础策略抽象类

所有策略必须继承 BaseStrategy 并实现 generate_signals() 方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Optional

import pandas as pd


class Signal(IntEnum):
    """交易信号枚举"""
    SELL = -1      # 卖出 / 做空
    HOLD = 0       # 持有 / 空仓
    BUY = 1        # 买入 / 做多


@dataclass
class StrategyResult:
    """策略执行结果"""
    name: str                          # 策略名称
    params: Dict[str, Any] = field(default_factory=dict)
    signals: Optional[pd.Series] = None       # 信号序列 (index=date)
    positions: Optional[pd.Series] = None     # 持仓序列 (0/1)
    equity_curve: Optional[pd.Series] = None  # 权益曲线
    metrics: Dict[str, float] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    量化策略抽象基类

    子类需要:
      1. 在 __init__ 中声明可调参数
      2. 实现 generate_signals(df) → pd.Series[Signal]
      3. 实现 description 属性
    """

    def __init__(self, name: str = "Base"):
        self.name = name
        self._params: Dict[str, Any] = {}

    # ── 子类必须实现 ──────────────────────────────

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        根据 OHLCV 数据生成交易信号。

        Args:
            df: 必须包含列: open, high, low, close, volume
                索引为日期 (DatetimeIndex)

        Returns:
            pd.Series[Signal]: 同 index，每行一个信号
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """策略简介（中文）"""
        ...

    # ── 公共方法 ──────────────────────────────────

    @property
    def params(self) -> Dict[str, Any]:
        """返回策略当前参数（只读副本）"""
        return dict(self._params)

    def set_params(self, **kwargs) -> "BaseStrategy":
        """批量修改参数，支持链式调用"""
        for k, v in kwargs.items():
            if k in self._params:
                self._params[k] = v
            else:
                raise KeyError(f"未知参数 '{k}'，可用参数: {list(self._params.keys())}")
        return self

    def run(self, df: pd.DataFrame) -> StrategyResult:
        """
        完整的策略执行入口：生成信号 + 信号→持仓映射。

        默认逻辑：信号为 BUY 时持仓=1，SELL 时持仓=0，
        遇到 T+1 延迟（当日信号次日执行）。

        子类可覆盖 run() 或 _signal_to_position()。
        """
        signals = self.generate_signals(df)
        positions = self._signal_to_position(signals)

        return StrategyResult(
            name=self.name,
            params=self.params,
            signals=signals,
            positions=positions,
        )

    def _signal_to_position(self, signals: pd.Series) -> pd.Series:
        """
        将信号转换为持仓 (0/1)，默认逻辑：
        - BUY 信号次日开盘建仓
        - SELL 信号次日开盘平仓
        - HOLD 维持前一状态
        """
        position = 0
        positions = pd.Series(0, index=signals.index, dtype=int)

        for i in range(len(signals)):
            sig = signals.iloc[i]
            if sig == Signal.BUY:
                position = 1
            elif sig == Signal.SELL:
                position = 0
            # HOLD → 维持
            positions.iloc[i] = position

        # T+1 延迟：信号实际在下一个交易日执行
        positions = positions.shift(1).fillna(0).astype(int)
        return positions

    # ── 指标计算辅助 ──────────────────────────────

    @staticmethod
    def ma(series: pd.Series, period: int) -> pd.Series:
        """简单移动平均"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """指数移动平均"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def std(series: pd.Series, period: int) -> pd.Series:
        """滚动标准差"""
        return series.rolling(window=period).std()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标 RSI"""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"
