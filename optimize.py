"""
参数优化 & 策略诊断工具

用法:
  python optimize.py --diagnose pb        # 诊断 PB 策略信号
  python optimize.py --optimize volume     # 优化放量突破策略
  python optimize.py --optimize pb         # 优化 PB 估值策略
  python optimize.py --all                 # 全部优化
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data_utils import fetch_boc_data
from strategies import STRATEGY_REGISTRY
from backtest_engine import BacktestEngine, BacktestResult


def diagnose_pb(df: pd.DataFrame):
    """诊断 PB 估值策略：分析价格分位数分布 & 信号触发位置"""
    lookback = 504
    close = df["close"]

    # 滚动分位数
    rolling_pct = close.rolling(window=lookback, min_periods=lookback // 2).rank(pct=True) * 100

    print(f"\n{'='*65}")
    print(f"  PB 估值策略诊断 (lookback={lookback}天)")
    print(f"{'='*65}")

    # 分位数分布统计
    valid = rolling_pct.dropna()
    print(f"\n  ── 滚动价格分位数分布 ──")
    print(f"  有效数据: {len(valid)} / {len(df)} 天")
    for p in [0, 5, 10, 20, 25, 50, 75, 80, 90, 95, 100]:
        val = np.percentile(valid, p)
        print(f"  P{p:3d}: {val:5.1f}")

    below_p20 = (valid <= 20).sum()
    above_p80 = (valid >= 80).sum()
    print(f"\n  分位 ≤ 20 (买入区): {below_p20} 天 ({below_p20/len(valid)*100:.1f}%)")
    print(f"  分位 ≥ 80 (卖出区): {above_p80} 天 ({above_p80/len(valid)*100:.1f}%)")

    # 按年份统计分位数均值
    print(f"\n  ── 逐年分位数均值 ──")
    yearly = valid.groupby(valid.index.year).agg(["mean", "min", "max"])
    for yr, row in yearly.iterrows():
        m, mi, mx = row.iloc[0], row.iloc[1], row.iloc[2]
        bar = "█" * int(m / 5) + "░" * (20 - int(m / 5))
        print(f"  {yr}: P50={m:4.1f} [{mi:4.1f}, {mx:4.1f}] {bar}")

    # 找买入信号触发的时间点
    from strategies.pb_valuation import PBValuation
    strategy = PBValuation()
    signals = strategy.generate_signals(df)

    buy_dates = df.index[signals == 1]
    sell_dates = df.index[signals == -1]
    print(f"\n  ── 信号历史 ──")
    print(f"  BUY 信号: {len(buy_dates)} 次")
    for d in buy_dates:
        pct_val = rolling_pct.loc[d] if d in rolling_pct.index else np.nan
        price = close.loc[d] if d in close.index else np.nan
        print(f"    {d.date()}  价格={price:.2f}  分位={pct_val:.1f}")

    print(f"  SELL 信号: {len(sell_dates)} 次")
    for d in sell_dates:
        pct_val = rolling_pct.loc[d] if d in rolling_pct.index else np.nan
        price = close.loc[d] if d in close.index else np.nan
        print(f"    {d.date()}  价格={price:.2f}  分位={pct_val:.1f}")


def optimize_strategy(strategy_name: str, df: pd.DataFrame,
                      param_grid: dict, metric: str = "total_return",
                      top_n: int = 10) -> pd.DataFrame:
    """网格搜索参数优化"""
    cls = STRATEGY_REGISTRY[strategy_name]
    engine = BacktestEngine()

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = 1
    for v in values:
        total *= len(v)

    print(f"\n{'='*65}")
    print(f"  优化: {cls().__class__.__name__}")
    print(f"  参数空间: {total} 组 | 指标: {metric}")
    print(f"{'='*65}")

    results = []
    count = 0
    for combo in itertools.product(*values):
        count += 1
        params = dict(zip(keys, combo))

        try:
            strategy = cls()
            strategy.set_params(**params)
            sr = strategy.run(df)
            bt = engine.run(df, sr.signals)

            results.append({
                **params,
                "total_return": bt.total_return,
                "annual_return": bt.annual_return,
                "sharpe": bt.sharpe_ratio,
                "max_dd": bt.max_drawdown,
                "win_rate": bt.win_rate,
                "trades": bt.total_trades,
                "profit_factor": bt.profit_factor,
            })
        except Exception as e:
            continue

        if count % 50 == 0:
            print(f"  ... {count}/{total} ({(count/total)*100:.0f}%)")

    print(f"  完成: {len(results)}/{total} 组有效")

    df_res = pd.DataFrame(results).sort_values(metric, ascending=False)
    return df_res.head(top_n)


# ── 预定义参数网格 ──────────────────────────────

GRID_VOLUME = {
    "price_period": [10, 20, 30, 50],
    "vol_period": [10, 20, 30],
    "vol_multiplier": [1.2, 1.5, 2.0],
    "exit_vol_shrink": [0.3, 0.5, 0.7],
    "hold_days_min": [1, 3, 5],
}

GRID_PB = {
    "lookback": [252, 504, 756, 1008],        # 1yr, 2yr, 3yr, 4yr
    "buy_percentile": [10, 15, 20, 25, 30],
    "sell_percentile": [70, 75, 80, 85, 90],
}

GRID_BOLLINGER = {
    "period": [10, 20, 30, 50],
    "k": [1.5, 2.0, 2.5],
    "stop_loss_margin": [0.02, 0.03, 0.05],
}

GRID_MA = {
    "fast": [3, 5, 10, 15],
    "slow": [15, 20, 30, 50],
    "use_adx": [True, False],
    "adx_threshold": [15, 20, 25],
}

OPTIMIZE_GRIDS = {
    "volume": GRID_VOLUME,
    "pb": GRID_PB,
    "bollinger": GRID_BOLLINGER,
    "ma_cross": GRID_MA,
}


# ── 入口 ────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="策略诊断 & 参数优化")
    p.add_argument("--diagnose", type=str, help="诊断指定策略 (pb)")
    p.add_argument("--optimize", type=str, help="优化指定策略 (volume, pb, bollinger, ma_cross)")
    p.add_argument("--all", action="store_true", help="优化所有可优化策略")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default="2026-07-31")
    p.add_argument("--top", type=int, default=10, help="显示前 N 组参数")
    args = p.parse_args()

    print(f"[数据] 加载数据: {args.start} ~ {args.end}")
    df = fetch_boc_data(args.start, args.end)
    print(f"[数据] {len(df)} 个交易日")

    if args.diagnose:
        if args.diagnose == "pb":
            diagnose_pb(df)
        else:
            print(f"暂不支持诊断: {args.diagnose}")

    if args.optimize:
        optimize_and_show(args.optimize, df, args.top)

    if args.all:
        for name in OPTIMIZE_GRIDS:
            optimize_and_show(name, df, args.top)


def optimize_and_show(name: str, df: pd.DataFrame, top_n: int):
    if name not in OPTIMIZE_GRIDS:
        print(f"未知策略 '{name}'。可优化: {list(OPTIMIZE_GRIDS.keys())}")
        return
    grid = OPTIMIZE_GRIDS[name]
    top = optimize_strategy(name, df, grid, top_n=top_n)
    print(f"\n  Top {min(len(top), top_n)} 参数组合:\n")
    print(top.to_string(float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))
    return top


if __name__ == "__main__":
    main()
