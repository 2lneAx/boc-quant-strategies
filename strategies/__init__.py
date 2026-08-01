"""
量化策略库 — 中国银行 (601988)

包含以下策略:
  - BollingerBreakout    : 布林带均值回归
  - RSIReversion         : RSI 超买超卖
  - MACross              : 双均线交叉
  - DonchianBreakout     : 唐奇安通道突破
  - VolumeBreakout       : 放量突破
  - OBVDivergence        : OBV 背离
  - DividendStrategy     : 股息分红策略
  - EarningsStrategy     : 财报季报效应
  - PairsTrading         : 银行股配对交易
  - IndexLinkage         : 上证50/沪深300 联动
  - PBValuation          : PB 估值区间择时
"""

from .base import BaseStrategy, Signal
from .bollinger import BollingerBreakout
from .rsi import RSIReversion
from .ma_cross import MACross
from .donchian import DonchianBreakout
from .volume_breakout import VolumeBreakout
from .obv_divergence import OBVDivergence
from .dividend import DividendStrategy
from .earnings import EarningsStrategy
from .pairs_trading import PairsTrading
from .index_linkage import IndexLinkage
from .pb_valuation import PBValuation
from .combo import ComboStrategy, print_combo_result
from .dynamic_alloc import DynamicAllocation, print_dynamic_result

# 策略注册表 — 按名称快速切换
STRATEGY_REGISTRY = {
    "bollinger": BollingerBreakout,
    "rsi": RSIReversion,
    "ma_cross": MACross,
    "donchian": DonchianBreakout,
    "volume": VolumeBreakout,
    "obv": OBVDivergence,
    "dividend": DividendStrategy,
    "earnings": EarningsStrategy,
    "pairs": PairsTrading,
    "index": IndexLinkage,
    "pb": PBValuation,
    "combo": ComboStrategy,
    "dynamic": DynamicAllocation,
}

__all__ = [
    "BaseStrategy", "Signal",
    "BollingerBreakout", "RSIReversion", "MACross",
    "DonchianBreakout", "VolumeBreakout", "OBVDivergence",
    "DividendStrategy", "EarningsStrategy",
    "PairsTrading", "IndexLinkage", "PBValuation",
    "STRATEGY_REGISTRY",
]
