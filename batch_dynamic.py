"""
动态仓位+股息再投资 — 跨行业批量测试

找出这个策略真正适用于哪类股票。
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import time

from data_utils import _ensure_cache_dir, _save_cache, _load_cache, _parse_akshare_df
from strategies.dynamic_alloc import DynamicAllocation, print_dynamic_result

# 测试标的
STOCKS = {
    # 银行股
    "601288": ("农业银行", "银行-四大行"),
    "601398": ("工商银行", "银行-四大行"),
    "601939": ("建设银行", "银行-四大行"),
    "601988": ("中国银行", "银行-四大行"),
    "601998": ("中信银行", "银行-股份制"),
    "600036": ("招商银行", "银行-零售"),
    # 高股息
    "600900": ("长江电力", "公用事业-水电"),
    "601088": ("中国神华", "能源-煤炭"),
    "600028": ("中国石化", "能源-石油"),
    # 成长/消费
    "600519": ("贵州茅台", "消费-白酒"),
    "000858": ("五粮液",   "消费-白酒"),
    # 指数ETF (用个股近似，真实ETF需不同数据源)
    # 保险
    "601318": ("中国平安", "金融-保险"),
}


def fetch_stock(symbol: str, start: str, end: str):
    cache_dir = _ensure_cache_dir()
    cache_file = cache_dir / f"stock_{symbol}.pkl"

    df = _load_cache(cache_file)
    if df is not None and not df.empty:
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy() if mask.any() else None

    try:
        import akshare as ak
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol, start_date=start.replace("-", ""),
            end_date=end.replace("-", ""), adjust="hfq"
        )
        if df is not None and len(df) > 0:
            df = _parse_akshare_df(df)
            _save_cache(cache_file, df)
            mask = (df.index >= start) & (df.index <= end)
            return df.loc[mask].copy() if mask.any() else None
    except Exception as e:
        print(f"  [{symbol}] 失败: {e}")
        return None


def test_stock(symbol, name, category, start, end):
    """测试动态仓位策略"""
    print(f"  [{symbol} {name}] ", end="", flush=True)
    df = fetch_stock(symbol, start, end)
    if df is None or df.empty:
        print("数据获取失败")
        return None

    print(f"{len(df)}天 ", end="", flush=True)

    strategy = DynamicAllocation()
    result = strategy.run(df, initial_capital=100_000)
    m = result.metrics

    bench_ret = m["benchmark_return"]
    combo_ret = m["total_return"]
    excess = combo_ret - bench_ret

    icon = "🟢" if excess > 0 else ("🟡" if excess > -0.3 else "🔴")
    print(f"{icon} 策略:{combo_ret:.1%} 基准:{bench_ret:.1%} "
          f"Sharpe:{m['sharpe_ratio']:.2f} MDD:{m['max_drawdown']:.1%} "
          f"股息:{m['total_dividends']:.0f}元 调仓:{m['trade_count']}次")

    return {
        "symbol": symbol,
        "name": name,
        "category": category,
        "days": len(df),
        "combo_ret": combo_ret,
        "bench_ret": bench_ret,
        "excess": excess,
        "sharpe": m["sharpe_ratio"],
        "max_dd": m["max_drawdown"],
        "dividends": m["total_dividends"],
        "trades": m["trade_count"],
        "avg_pos": m["avg_position"],
    }


def main():
    start = "2010-01-01"
    end = "2026-07-31"

    print(f"\n{'='*80}")
    print(f"  动态仓位+股息再投资 — 跨行业测试")
    print(f"  区间: {start} ~ {end}")
    print(f"{'='*80}\n")

    results = []
    for symbol, (name, category) in STOCKS.items():
        r = test_stock(symbol, name, category, start, end)
        if r:
            results.append(r)
        time.sleep(0.5)

    # ── 排名 ──────────────────────────────
    print(f"\n{'='*80}")
    print(f"  按超额收益排名")
    print(f"{'='*80}\n")
    print(f"{'股票':10s} {'类型':14s} {'天数':>5s} {'策略':>8s} {'基准':>8s} {'超额':>8s} {'夏普':>6s} {'回撤':>7s} {'股息':>8s} {'调仓':>4s}")
    print(f"{'-'*10} {'-'*14} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*4}")

    for r in sorted(results, key=lambda x: x["excess"], reverse=True):
        print(f"{r['name']:10s} {r['category']:14s} {r['days']:5d} "
              f"{r['combo_ret']:7.1%} {r['bench_ret']:7.1%} {r['excess']:+7.1%} "
              f"{r['sharpe']:5.2f} {r['max_dd']:6.1%} {r['dividends']:7.0f}元 {r['trades']:4d}")

    # ── 按夏普排名 ────────────────────────
    print(f"\n{'='*80}")
    print(f"  按夏普比率排名")
    print(f"{'='*80}\n")
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"  {r['name']:10s} Sharpe={r['sharpe']:.2f}  "
              f"收益={r['combo_ret']:.1%}  超额={r['excess']:+.1%}  "
              f"回撤={r['max_dd']:.1%}  股息={r['dividends']:.0f}元")


if __name__ == "__main__":
    main()
