"""
银行股批量回测 — Combo 策略 (Vol 50% + PB 50%)

用法:
  python batch_test.py            # 默认6只银行股
  python batch_test.py --all      # 全部可测银行股
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import time

from data_utils import _fetch_real_data, _parse_akshare_df, _ensure_cache_dir, _save_cache, _load_cache
from strategies.combo import ComboStrategy

BANKS = {
    "601988": "中国银行",
    "601398": "工商银行",
    "601939": "建设银行",
    "601288": "农业银行",
    "601998": "中信银行",
    "601229": "上海银行",
}


def fetch_bank_data(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """获取指定银行的后复权日线"""
    cache_dir = _ensure_cache_dir()
    cache_file = cache_dir / f"bank_{symbol}.pkl"

    # 缓存
    df = _load_cache(cache_file)
    if df is not None and not df.empty:
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy() if mask.any() else None

    # 获取
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="hfq"
        )
        if df is not None and len(df) > 0:
            df = _parse_akshare_df(df)
            _save_cache(cache_file, df)
            mask = (df.index >= start) & (df.index <= end)
            return df.loc[mask].copy() if mask.any() else None
    except Exception as e:
        print(f"  [{symbol}] 获取失败: {e}")
        return None


def run_combo_on_bank(symbol: str, name: str, start: str, end: str):
    """对一只银行股运行 Combo 策略"""
    print(f"  [{symbol} {name}] 获取数据...", end=" ")
    df = fetch_bank_data(symbol, start, end)
    if df is None or df.empty:
        print("FAIL")
        return None

    print(f"{len(df)} 天", end=" ")

    strategy = ComboStrategy()
    result = strategy.run(df, initial_capital=100_000)

    m = result.metrics
    bench_ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    print(f"→ 组合:{m['total_return']:.1%} 基准:{bench_ret:.1%} "
          f"Sharpe:{m['sharpe_ratio']:.2f} MDD:{m['max_drawdown']:.1%}")

    return {
        "symbol": symbol,
        "name": name,
        "days": len(df),
        "combo_ret": m["total_return"],
        "bench_ret": bench_ret,
        "excess": m["total_return"] - bench_ret,
        "sharpe": m["sharpe_ratio"],
        "max_dd": m["max_drawdown"],
        "vol_trades": m["vol_trades"],
        "pb_trades": m["pb_trades"],
        "start": df.index[0].date(),
        "end": df.index[-1].date(),
    }


def main():
    start = "2010-01-01"
    end = "2026-07-31"

    print(f"\n{'='*75}")
    print(f"  银行股 Combo 策略批量回测 (Vol 50% + PB 50%)")
    print(f"  回测区间: {start} ~ {end}")
    print(f"{'='*75}\n")

    results = []
    for symbol, name in BANKS.items():
        r = run_combo_on_bank(symbol, name, start, end)
        if r:
            results.append(r)
        time.sleep(1)  # 避免 API 限流

    # ── 排名 ──────────────────────────────
    print(f"\n{'='*75}")
    print(f"  排名对比")
    print(f"{'='*75}\n")
    print(f"{'银行':8s} {'数据':>5s} {'区间':22s} {'组合收益':>9s} {'基准收益':>9s} {'超额':>8s} {'夏普':>6s} {'回撤':>7s} {'Vol':>4s} {'PB':>4s}")
    print(f"{'-'*8} {'-'*5} {'-'*22} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*7} {'-'*4} {'-'*4}")

    for r in sorted(results, key=lambda x: x["combo_ret"], reverse=True):
        date_range = f"{r['start']}~{r['end']}"
        print(f"{r['name']:8s} {r['days']:5d} {date_range:22s} "
              f"{r['combo_ret']:8.1%} {r['bench_ret']:8.1%} {r['excess']:+7.1%} "
              f"{r['sharpe']:5.2f} {r['max_dd']:6.1%} {r['vol_trades']:4d} {r['pb_trades']:4d}")


if __name__ == "__main__":
    main()
