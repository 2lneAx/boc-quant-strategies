"""
回测引擎

功能:
  - 根据策略信号模拟交易
  - 计算收益曲线和核心指标
  - T+1 交易规则（当日信号 → 次日开盘执行）
  - 万元单位手续费
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0          # 总收益率
    annual_return: float = 0.0         # 年化收益率
    annual_volatility: float = 0.0     # 年化波动率
    sharpe_ratio: float = 0.0          # 夏普比率
    max_drawdown: float = 0.0          # 最大回撤
    max_drawdown_duration: int = 0     # 最长回撤持续天数
    win_rate: float = 0.0              # 胜率（交易次数中盈利占比）
    profit_factor: float = 0.0         # 盈亏比
    total_trades: int = 0              # 总交易次数
    avg_trade_return: float = 0.0      # 平均每笔收益率
    benchmark_return: float = 0.0      # 基准收益（买入持有）
    equity_curve: Optional[pd.Series] = None       # 策略权益曲线
    benchmark_curve: Optional[pd.Series] = None    # 基准权益曲线
    trade_log: Optional[pd.DataFrame] = None       # 交易记录


class BacktestEngine:
    """
    向量化回测引擎。

    交易规则:
      - T+1: 当日信号在次日开盘价执行
      - 手续费: 万分之三 (双边)
      - 印花税: 千分之一 (卖出单边)
      - 初始资金: 100,000 元
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.0003,   # 万三
        stamp_tax_rate: float = 0.001,     # 千一（仅卖出）
        slippage: float = 0.001,           # 滑点 0.1%
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage

    def run(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """
        执行回测。

        Args:
            df      : OHLCV 数据，DatetimeIndex
            signals : 交易信号序列，同 index, 1=buy, -1=sell, 0=hold

        Returns:
            BacktestResult 含所有指标和曲线
        """
        close = df["close"].values
        n = len(close)

        # ── 权益曲线 ──────────────────────────
        cash = self.initial_capital
        shares = 0
        equity = np.full(n, self.initial_capital)  # 初始化为初始资金
        position_flag = np.zeros(n, dtype=int)  # 0=空仓, 1=持仓

        # 交易记录
        trades = []

        for i in range(1, n):  # 从第1天开始（需要前一天信号）
            sig = signals.iloc[i - 1]  # 前一日信号，今日执行

            # 当日执行价 = 开盘价（加滑点）
            exec_price = df["open"].iloc[i]  # 今日开盘

            if sig == 1 and shares == 0:  # 买入信号 + 空仓
                # 计算可买股数（100股整数倍）
                cost_per_share = exec_price * (1 + self.slippage)
                max_shares = int(cash / (cost_per_share * (1 + self.commission_rate)) / 100) * 100
                if max_shares >= 100:
                    total_cost = max_shares * cost_per_share * (1 + self.commission_rate)
                    if total_cost <= cash:
                        cash -= total_cost
                        shares = max_shares
                        position_flag[i] = 1
                        trades.append({
                            "date": df.index[i], "action": "BUY",
                            "price": exec_price, "shares": shares,
                            "cost": total_cost, "cash_after": cash,
                        })

            elif sig == -1 and shares > 0:  # 卖出信号 + 持仓
                sell_price = exec_price * (1 - self.slippage)
                proceeds = shares * sell_price
                commission = proceeds * self.commission_rate
                stamp_tax = proceeds * self.stamp_tax_rate
                cash += proceeds - commission - stamp_tax
                trades.append({
                    "date": df.index[i], "action": "SELL",
                    "price": exec_price, "shares": shares,
                    "proceeds": proceeds, "cash_after": cash,
                })
                shares = 0
                position_flag[i] = 0
            else:
                position_flag[i] = 1 if shares > 0 else 0

            # 当日权益 = 现金 + 持仓市值
            equity[i] = cash + shares * close[i]

        # 最后一日若仍持仓，按收盘价平仓（不计税费）
        if shares > 0:
            equity[-1] = cash + shares * close[-1]

        equity_series = pd.Series(equity, index=df.index)

        # ── 基准曲线（买入持有） ─────────────
        bench_shares = int(self.initial_capital / (df["open"].iloc[0] * 1.001) / 100) * 100
        bench_cost = bench_shares * df["open"].iloc[0] * (1 + self.commission_rate)
        bench_equity = (bench_shares * close) + (self.initial_capital - bench_cost)
        bench_series = pd.Series(bench_equity, index=df.index)

        # ── 指标计算 ────────────────────────
        result = BacktestResult()
        result.equity_curve = equity_series
        result.benchmark_curve = bench_series
        result.trade_log = pd.DataFrame(trades) if trades else pd.DataFrame()

        if len(equity_series) > 1:
            result = self._compute_metrics(equity_series, bench_series, trades, df)

        return result

    def _compute_metrics(
        self, equity: pd.Series, bench: pd.Series,
        trades: list, df: pd.DataFrame
    ) -> BacktestResult:
        """计算所有回测指标"""
        result = BacktestResult()
        result.equity_curve = equity
        result.benchmark_curve = bench
        result.trade_log = pd.DataFrame(trades) if trades else pd.DataFrame()

        n_days = len(equity)
        trading_years = n_days / 252

        # ── 总收益 ──────────────────────────
        final_equity = equity.iloc[-1]
        result.total_return = (final_equity - self.initial_capital) / self.initial_capital
        result.benchmark_return = (bench.iloc[-1] - self.initial_capital) / self.initial_capital

        # ── 年化收益 / 波动 ──────────────────
        daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            result.annual_return = (1 + result.total_return) ** (1 / max(trading_years, 0.5)) - 1
            result.annual_volatility = daily_returns.std() * np.sqrt(252)
            rf_daily = 0.025 / 252  # 无风险利率 ~2.5%
            excess = daily_returns - rf_daily
            result.sharpe_ratio = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0

        # ── 最大回撤 ────────────────────────
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        result.max_drawdown = drawdown.min()

        # 最长回撤持续天数
        dd_start = None
        max_dd_days = 0
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0 and dd_start is None:
                dd_start = i
            elif drawdown.iloc[i] >= 0 and dd_start is not None:
                max_dd_days = max(max_dd_days, i - dd_start)
                dd_start = None
        if dd_start is not None:
            max_dd_days = max(max_dd_days, len(drawdown) - dd_start)
        result.max_drawdown_duration = max_dd_days

        # ── 交易统计 ────────────────────────
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        if trades and sell_trades:
            # 配对买入-卖出计算每笔收益
            buy_trades = [t for t in trades if t["action"] == "BUY"]
            result.total_trades = min(len(buy_trades), len(sell_trades))
            trade_returns = []
            for j in range(result.total_trades):
                buy = buy_trades[j]
                sell = sell_trades[j]
                ret = (sell["proceeds"] - buy["cost"]) / buy["cost"]
                trade_returns.append(ret)

            if trade_returns:
                result.win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
                result.avg_trade_return = np.mean(trade_returns)
                total_profit = sum(r for r in trade_returns if r > 0)
                total_loss = abs(sum(r for r in trade_returns if r < 0))
                result.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        return result


def print_result(result: BacktestResult, strategy_name: str = "") -> None:
    """格式化打印回测结果"""
    print(f"\n{'='*60}")
    print(f"  回测报告: {strategy_name}")
    print(f"{'='*60}")
    print(f"  总收益率      : {result.total_return:>10.2%}")
    print(f"  年化收益率    : {result.annual_return:>10.2%}")
    print(f"  年化波动率    : {result.annual_volatility:>10.2%}")
    print(f"  夏普比率      : {result.sharpe_ratio:>10.2f}")
    print(f"  最大回撤      : {result.max_drawdown:>10.2%}")
    print(f"  最长回撤天数  : {result.max_drawdown_duration:>10}")
    print(f"  胜率          : {result.win_rate:>10.2%}")
    print(f"  盈亏比        : {result.profit_factor:>10.2f}")
    print(f"  总交易次数    : {result.total_trades:>10}")
    print(f"  平均每笔收益  : {result.avg_trade_return:>10.2%}")
    print(f"  ─────────────────────────────────")
    print(f"  基准收益(持有): {result.benchmark_return:>10.2%}")
    print(f"  超额收益      : {result.total_return - result.benchmark_return:>10.2%}")
    print(f"{'='*60}\n")
