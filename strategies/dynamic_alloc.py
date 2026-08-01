"""
动态仓位 + 股息再投资策略

核心理念:
  永远不空仓、永远不满仓。根据估值分位数在 20%~80% 仓位之间动态调整。
  收到股息后立即再投资，让复利持续发挥作用。

与之前 Combo 的关键区别:
  1) 连续仓位 (20-80%) vs 二元仓位 (0/1)
  2) 股息显式再投资 vs 后复权隐含股息
  3) 永远在场 vs 可能踏空

参数:
    lookback      : PB分位窗口 (默认 252)
    min_position  : 最低仓位 (默认 0.20，即永不满仓低于20%)
    max_position  : 最高仓位 (默认 0.80，即永远保留20%现金)
"""

import pandas as pd
import numpy as np

from .base import BaseStrategy, Signal, StrategyResult


class DynamicAllocation(BaseStrategy):
    """动态仓位 + 股息再投资"""

    def __init__(self):
        super().__init__(name="动态分配+股息再投")
        self._params = {
            "lookback": 252,
            "min_position": 0.20,
            "max_position": 0.80,
        }

    @property
    def description(self) -> str:
        lb = self._params["lookback"]
        lo = self._params["min_position"]
        hi = self._params["max_position"]
        return f"动态仓位 (lookback={lb}d, {lo:.0%}-{hi:.0%}) + 股息再投"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(Signal.HOLD, index=df.index, dtype=int)

    def run(self, df: pd.DataFrame, initial_capital: float = 100_000.0) -> StrategyResult:
        lookback = self._params["lookback"]
        min_pos = self._params["min_position"]
        max_pos = self._params["max_position"]

        close = df["close"].values
        open_p = df["open"].values
        n = len(close)

        # ── 估值分位数 ──────────────────────────
        rolling_pct = pd.Series(close).rolling(window=lookback, min_periods=lookback // 2)
        rank_pct = rolling_pct.rank(pct=True).fillna(0.5).values * 100

        # ── 计算目标仓位 ────────────────────────
        # 分位数 100=极度高估→min仓位, 0=极度低估→max仓位
        # target_pos = max_pos - (max_pos - min_pos) * (rank_pct / 100)
        target_positions = max_pos - (max_pos - min_pos) * (rank_pct / 100.0)
        target_positions = np.clip(target_positions, min_pos, max_pos)

        # ── 股息日期检测 ────────────────────────
        # 后复权数据: 除权日当天 hfq 价格会跳升（调整因子 > 1）
        # adj_factor[i] = hfq_close[i] / hfq_close[i-1] vs actual_close[i] / actual_close[i-1]
        # 简化: 用振幅异常检测
        # 实际方法: 用 pct_change 检测单日异常正向跳动
        daily_ret = pd.Series(close).pct_change().values
        # 股息通常导致 hfq 价格单日跳升 1-5%
        # 找正向异常（排除正常波动）
        median_ret = np.nanmedian(daily_ret)
        std_ret = np.nanstd(daily_ret)
        div_threshold = median_ret + 2.5 * std_ret
        is_div_day = (daily_ret > div_threshold) & (daily_ret > 0.01)

        # 估计每股股息 = 跳升幅度 × 前一天收盘价
        estimated_dividends = np.zeros(n)
        for i in range(1, n):
            if is_div_day[i] and not np.isnan(daily_ret[i]):
                estimated_dividends[i] = close[i-1] * daily_ret[i] * 0.9  # 90%归因于股息

        # ── 模拟交易 ────────────────────────────
        cash = initial_capital * (1 - target_positions[0])
        shares = (initial_capital * target_positions[0]) / open_p[0]
        # 取整到100股
        shares = int(shares / 100) * 100
        cash = initial_capital - shares * open_p[0] * 1.002

        equity = np.zeros(n)
        equity[0] = cash + shares * close[0]

        actual_positions = np.zeros(n)
        actual_positions[0] = (shares * close[0]) / equity[0] if equity[0] > 0 else 0

        total_dividends = 0.0
        trade_count = 0

        for i in range(1, n):
            cp = close[i]
            op = open_p[i]

            # ── 股息入账 + 再投资 ────────────────
            if estimated_dividends[i] > 0 and shares > 0:
                div_cash = shares * estimated_dividends[i]
                total_dividends += div_cash
                cash += div_cash
                # 股息立刻再投资 → 买更多股票
                if op > 0:
                    extra_shares = int(div_cash / (op * 1.002) / 100) * 100
                    if extra_shares >= 100:
                        cost = extra_shares * op * 1.002
                        cash -= cost
                        shares += extra_shares

            # ── 调仓到目标仓位 ────────────────────
            target_pos = target_positions[i]
            current_equity = cash + shares * cp
            target_stock_value = current_equity * target_pos
            current_stock_value = shares * cp
            diff = target_stock_value - current_stock_value

            # 调仓阈值: 偏离超过 5% 才操作（减少摩擦）
            if abs(diff) > current_equity * 0.05:
                if diff > 0:  # 加仓
                    buy_value = min(diff, cash * 0.95)  # 留 5% 现金
                    add_shares = int(buy_value / (op * 1.002) / 100) * 100
                    if add_shares >= 100:
                        cost = add_shares * op * 1.002
                        cash -= cost
                        shares += add_shares
                        trade_count += 1
                else:  # 减仓
                    sell_value = min(-diff, shares * cp * 0.95)
                    sell_shares = int(sell_value / cp / 100) * 100
                    if sell_shares >= 100:
                        proceeds = sell_shares * op * 0.998
                        cash += proceeds
                        shares -= sell_shares
                        trade_count += 1

            actual_positions[i] = (shares * cp) / (cash + shares * cp) if (cash + shares * cp) > 0 else 0
            equity[i] = cash + shares * cp

        # ── 指标 ────────────────────────────────
        total_ret = (equity[-1] - initial_capital) / initial_capital
        yrs = n / 252
        ann_ret = (1 + total_ret) ** (1 / yrs) - 1

        eq_s = pd.Series(equity, index=df.index)
        daily_ret_s = eq_s.pct_change().dropna()
        ann_vol = daily_ret_s.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.025) / ann_vol if ann_vol > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()

        bench_ret = (close[-1] / close[0] - 1)

        metrics = {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "benchmark_return": bench_ret,
            "total_dividends": total_dividends,
            "trade_count": trade_count,
            "avg_position": np.mean(actual_positions),
            "final_position": actual_positions[-1],
        }

        return StrategyResult(
            name=self.name,
            params=self.params,
            equity_curve=eq_s,
            metrics=metrics,
        )


def print_dynamic_result(result: StrategyResult) -> None:
    """打印动态仓位策略结果"""
    m = result.metrics

    print(f"\n{'='*65}")
    print(f"  动态仓位 + 股息再投资策略")
    print(f"{'='*65}")
    print(f"  总收益率      : {m['total_return']:>10.2%}")
    print(f"  年化收益率    : {m['annual_return']:>10.2%}")
    print(f"  夏普比率      : {m['sharpe_ratio']:>10.2f}")
    print(f"  最大回撤      : {m['max_drawdown']:>10.2%}")
    print(f"  ─────────────────────────────────")
    print(f"  累计股息收入  : {m['total_dividends']:>10.0f} 元")
    print(f"  调仓次数      : {m['trade_count']:>10} 次")
    print(f"  平均仓位      : {m['avg_position']:>10.0%}")
    print(f"  最终仓位      : {m['final_position']:>10.0%}")
    print(f"  ─────────────────────────────────")
    print(f"  基准收益(持有): {m['benchmark_return']:>10.2%}")
    print(f"  超额收益      : {m['total_return'] - m['benchmark_return']:>10.2%}")
    print(f"{'='*65}\n")
