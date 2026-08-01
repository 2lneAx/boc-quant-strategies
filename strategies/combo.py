"""
双策略组合 (Combo) — Volume Breakout + PB Valuation

默认 55 开，两个策略各管一半资金，完全独立运行。
这个策略直接计算组合权益曲线，不走标准回测引擎
（因为涉及分仓，标准引擎只支持 0/1 仓位）。

参数:
    vol_weight : Volume 策略的资金权重 (默认 0.5)
    pb_weight  : PB 策略的资金权重 (默认 0.5)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, StrategyResult
from .volume_breakout import VolumeBreakout
from .pb_valuation import PBValuation


class ComboStrategy(BaseStrategy):
    """
    Volume + PB 双策略组合。

    两个策略各管理独立资金池，互不干扰。
    天然实现动态仓位管理——某个策略空仓时其资金就是现金。
    """

    def __init__(self):
        super().__init__(name="双策略组合")
        self._params = {
            "vol_weight": 0.50,   # Volume 资金权重
            "pb_weight": 0.50,    # PB 资金权重
        }

    @property
    def description(self) -> str:
        vw = self._params["vol_weight"]
        pw = self._params["pb_weight"]
        return f"组合 Vol({vw:.0%}) + PB({pw:.0%})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        组合策略不产生单一信号。
        真正的信号生成在 run() 中直接完成。
        """
        return pd.Series(Signal.HOLD, index=df.index, dtype=int)

    def run(self, df: pd.DataFrame, initial_capital: float = 100_000.0) -> StrategyResult:
        """直接计算组合权益曲线，不走标准回测引擎"""
        vol_w = self._params["vol_weight"]
        pb_w = self._params["pb_weight"]

        # 1) 运行两个子策略
        vol = VolumeBreakout()
        pb = PBValuation()
        sr_vol = vol.run(df)
        sr_pb = pb.run(df)

        pos_vol = sr_vol.positions  # 已含 T+1 shift
        pos_pb = sr_pb.positions

        close = df["close"].values
        open_p = df["open"].values
        n = len(close)

        # 2) 模拟双账户独立交易
        cash_vol = initial_capital * vol_w
        cash_pb = initial_capital * pb_w
        shares_vol = 0
        shares_pb = 0

        equity = np.full(n, initial_capital)
        trades = []

        for i in range(1, n):
            op = open_p[i]
            cp = close[i]

            # --- Volume 子账户 ---
            target = (pos_vol.iloc[i] == 1)
            if target and shares_vol == 0 and cash_vol > 0:
                max_s = int(cash_vol / (op * 1.002) / 100) * 100
                if max_s >= 100:
                    cost = max_s * op * 1.002
                    if cost <= cash_vol:
                        cash_vol -= cost
                        shares_vol = max_s
                        trades.append({
                            "date": df.index[i], "strategy": "Volume",
                            "action": "BUY", "price": op, "shares": max_s,
                        })
            elif not target and shares_vol > 0:
                cash_vol += shares_vol * op * 0.998
                trades.append({
                    "date": df.index[i], "strategy": "Volume",
                    "action": "SELL", "price": op, "shares": shares_vol,
                })
                shares_vol = 0

            # --- PB 子账户 ---
            target = (pos_pb.iloc[i] == 1)
            if target and shares_pb == 0 and cash_pb > 0:
                max_s = int(cash_pb / (op * 1.002) / 100) * 100
                if max_s >= 100:
                    cost = max_s * op * 1.002
                    if cost <= cash_pb:
                        cash_pb -= cost
                        shares_pb = max_s
                        trades.append({
                            "date": df.index[i], "strategy": "PB",
                            "action": "BUY", "price": op, "shares": max_s,
                        })
            elif not target and shares_pb > 0:
                cash_pb += shares_pb * op * 0.998
                trades.append({
                    "date": df.index[i], "strategy": "PB",
                    "action": "SELL", "price": op, "shares": shares_pb,
                })
                shares_pb = 0

            equity[i] = (cash_vol + shares_vol * cp +
                         cash_pb + shares_pb * cp)

        # 3) 计算指标
        total_ret = (equity[-1] - initial_capital) / initial_capital
        yrs = n / 252
        ann_ret = (1 + total_ret) ** (1 / yrs) - 1

        equity_s = pd.Series(equity, index=df.index)
        daily_ret = equity_s.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.025) / ann_vol if ann_vol > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()

        # 交易统计
        vol_trades = [t for t in trades if t["strategy"] == "Volume"]
        pb_trades = [t for t in trades if t["strategy"] == "PB"]
        vol_sells = [t for t in vol_trades if t["action"] == "SELL"]
        pb_sells = [t for t in pb_trades if t["action"] == "SELL"]

        # 4) 返回结果
        metrics = {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "vol_trades": len(vol_sells),
            "pb_trades": len(pb_sells),
            "total_trades": len(vol_sells) + len(pb_sells),
            "benchmark_return": (close[-1] / close[0] - 1),
        }

        return StrategyResult(
            name=self.name,
            params=self.params,
            equity_curve=equity_s,
            metrics=metrics,
        )


def print_combo_result(result: StrategyResult) -> None:
    """格式化打印组合回测结果"""
    m = result.metrics
    vw = result.params.get("vol_weight", 0.5)
    pw = result.params.get("pb_weight", 0.5)

    print(f"\n{'='*60}")
    print(f"  组合策略回测: Vol({vw:.0%}) + PB({pw:.0%})")
    print(f"{'='*60}")
    print(f"  总收益率      : {m['total_return']:>10.2%}")
    print(f"  年化收益率    : {m['annual_return']:>10.2%}")
    print(f"  夏普比率      : {m['sharpe_ratio']:>10.2f}")
    print(f"  最大回撤      : {m['max_drawdown']:>10.2%}")
    print(f"  ─────────────────────────────────")
    print(f"  Volume 交易   : {m['vol_trades']:>10} 笔")
    print(f"  PB 交易       : {m['pb_trades']:>10} 笔")
    print(f"  总交易次数    : {m['total_trades']:>10} 笔")
    print(f"  ─────────────────────────────────")
    print(f"  基准收益(持有): {m['benchmark_return']:>10.2%}")
    print(f"  超额收益      : {m['total_return'] - m['benchmark_return']:>10.2%}")
    print(f"{'='*60}\n")
