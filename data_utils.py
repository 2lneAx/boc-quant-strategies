"""
数据获取工具 — 中国银行 (601988)

支持 akshare 和本地缓存，如果 akshare 不可用则使用模拟数据回退。
"""

import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# ── 配置 ──────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"
SYMBOL = "601988"          # 中国银行
SYMBOL_NAME = "中国银行"
INDEX_SYMBOL = "000016"    # 上证50


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


# ── 主数据获取 ────────────────────────────────────

def fetch_boc_data(start: str = "2015-01-01", end: str = "2025-12-31",
                   use_cache: bool = True) -> pd.DataFrame:
    """
    获取中国银行 (601988) 日线数据。

    优先从本地缓存读取，其次尝试 akshare，都失败则生成模拟数据。

    Returns:
        DataFrame with columns: open, high, low, close, volume, turnover, amplitude, pct_change
        DatetimeIndex, 按日期升序排列
    """
    cache_file = _ensure_cache_dir() / f"boc_{SYMBOL}.pkl"

    # 1) 本地缓存
    if use_cache and cache_file.exists():
        df = _load_cache(cache_file)
        if df is not None and not df.empty:
            print(f"[数据] 从缓存加载: {cache_file}")
            return _filter_range(df, start, end)

    # 2) 真实数据源
    try:
        df = _fetch_real_data(start, end)
        if df is not None and not df.empty:
            _save_cache(cache_file, df)
            print(f"[数据] 真实数据获取成功，样本数: {len(df)}")
            return df
    except Exception as e:
        print(f"[数据] 真实数据获取失败: {e}")

    # 3) 模拟数据回退
    print("[数据] 使用模拟数据（回退方案）")
    df = _generate_mock_data(start, end)
    return df


def fetch_index_data(symbol: str = "000016", start: str = "2015-01-01",
                     end: str = "2025-12-31") -> pd.DataFrame:
    """
    获取指数数据（上证50: 000016, 沪深300: 000300）。

    Returns:
        DataFrame with columns: open, high, low, close, volume, amplitude, pct_change
    """
    cache_file = _ensure_cache_dir() / f"index_{symbol}.pkl"

    if cache_file.exists():
        df = _load_cache(cache_file)
        if df is not None and not df.empty:
            return _filter_range(df, start, end)

    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol=f"sh{symbol}")
        df.rename(columns={
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
        df["pct_change"] = df["close"].pct_change() * 100
        _save_cache(cache_file, df)
        return _filter_range(df, start, end)
    except Exception:
        return _generate_mock_data(start, end)


def fetch_bank_peers(start: str = "2015-01-01", end: str = "2025-12-31") -> dict:
    """
    获取银行股同行业数据（工行、建行、农行）。

    Returns:
        {symbol: DataFrame} 字典
    """
    peers = {
        "601398": "工商银行",
        "601939": "建设银行",
        "601288": "农业银行",
    }
    result = {}
    for sym, name in peers.items():
        cache_file = _ensure_cache_dir() / f"boc_{sym}.pkl"
        if cache_file.exists():
            df = _load_cache(cache_file)
        else:
            try:
                import akshare as ak
                df = ak.stock_zh_a_hist(symbol=sym, period="daily",
                                        start_date=start.replace("-", ""),
                                        end_date=end.replace("-", ""),
                                        adjust="qfq")
                df["日期"] = pd.to_datetime(df["日期"])
                df.set_index("日期", inplace=True)
                df.sort_index(inplace=True)
                df = df.rename(columns={c: cn_to_en(c) for c in df.columns})
                _save_cache(cache_file, df)
            except Exception:
                df = _generate_mock_data(start, end)
        result[sym] = _filter_range(df, start, end)
    return result


# ── 辅助函数 ──────────────────────────────────────

def _fetch_real_data(start: str, end: str) -> pd.DataFrame:
    """
    获取中国银行 (601988) 前复权日线。

    数据源优先级:
      1) akshare 腾讯源 (stock_zh_a_hist_tx) — 稳定，完整历史
      2) akshare 东方财富源 (stock_zh_a_hist) — 范围最全
      3) 直接腾讯 API — 仅 ~640 条，作为最后回退
    """
    import time

    start_d = start.replace("-", "")
    end_d = end.replace("-", "")

    # ── 方案 1: akshare 腾讯源 ───────────────
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist_tx(
            symbol=SYMBOL,
            start_date=start_d,
            end_date=end_d,
            adjust="hfq"  # 后复权：历史价格=真实成交价，保证始终为正
        )
        if df is not None and len(df) > 0:
            return _parse_akshare_df(df)
    except Exception as e:
        print(f"[数据] akshare(tx) 失败: {e}")

    # ── 方案 2: akshare 东方财富源 ───────────
    for attempt in range(2):
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=SYMBOL, period="daily",
                start_date=start_d, end_date=end_d,
                adjust="hfq"
            )
            if df is not None and len(df) > 0:
                return _parse_akshare_df(df)
        except Exception as e:
            if attempt < 1:
                time.sleep(5)

    # ── 方案 3: 直接腾讯 API ────────────────
    print("[数据] akshare 全部失败，尝试直连腾讯 API...")
    df_tx = _fetch_from_tencent(start, end)
    if df_tx is not None and len(df_tx) > 0:
        return df_tx

    raise RuntimeError("所有数据源均失败")


def _parse_akshare_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一解析 akshare 返回的 DataFrame（支持中英文列名）"""
    # 检测列名语言
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df.set_index("日期", inplace=True)
        df = df.rename(columns={c: cn_to_en(c) for c in df.columns})
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        # 尝试第一列作为日期
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col])
        df.set_index(first_col, inplace=True)

    df.sort_index(inplace=True)

    # 确保关键列存在
    for col, default in [("open", "close"), ("high", "close"), ("low", "close"),
                          ("volume", 0), ("turnover", 0),
                          ("amplitude", 0), ("pct_change", 0)]:
        if col not in df.columns:
            if default == "close":
                df[col] = df["close"]
            else:
                df[col] = default

    return df


def _fetch_from_tencent(start: str, end: str) -> pd.DataFrame | None:
    """
    直接从腾讯 API 抓取前复权日线。

    API: web.ifzq.gtimg.cn/appstock/app/fqkline/get

    每行格式: [date, open, close, high, low, volume]
    """
    import requests as _requests

    session = _requests.Session()
    session.trust_env = False  # 绕过 Windows 系统代理

    # 计算需要的 bar 数量（交易日 ≈ 自然日 * 0.7）
    days_estimate = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    count = max(int(days_estimate * 0.75), 100)

    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param=sh{SYMBOL},day,{start},,{count},hfq"
    )

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            return None

        stock_data = data.get("data", {}).get(f"sh{SYMBOL}", {})
        rows = stock_data.get("qfqday", []) or stock_data.get("day", [])

        if not rows:
            return None

        # 解析: [date, open, close, high, low, volume]
        records = []
        for row in rows:
            records.append({
                "date": pd.Timestamp(row[0]),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # 按日期范围裁剪
        df = df.loc[start:end].copy()

        if df.empty:
            return None

        # 补充衍生字段
        df["turnover"] = df["volume"] * df["close"] * 0.8  # 估算
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
        df["pct_change"] = df["close"].pct_change() * 100

        return df

    except Exception:
        return None


def cn_to_en(cn_name: str) -> str:
    """中文列名 → 英文列名"""
    mapping = {
        "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "turnover", "振幅": "amplitude",
        "涨跌幅": "pct_change", "换手率": "turnover_rate",
    }
    return mapping.get(cn_name, cn_name)


def _generate_mock_data(start: str, end: str) -> pd.DataFrame:
    """
    生成模拟日线数据（当真实数据源不可用时的回退方案）。

    模拟中国银行的走势特征：
    - 价格中枢 ~4.5 元
    - 年化波动 ~20%
    - 日均成交 ~1.5亿股
    - 包含若干个牛熊周期
    """
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    np.random.seed(42)

    # 构造带趋势+周期的价格路径
    t = np.arange(n) / 252  # 年化时间轴
    # 长期趋势 + 周期
    trend = 4.0 + 0.5 * np.sin(t * np.pi / 3) + 0.2 * t
    # 日收益率
    daily_ret = np.random.normal(0, 0.014, n)  # ~22% 年化波动
    price_path = trend * np.exp(np.cumsum(daily_ret))

    # 生成 OHLC
    close = price_path
    daily_range = close * 0.02  # 日内振幅约 2%
    high = close + np.abs(np.random.normal(0, daily_range, n))
    low = close - np.abs(np.random.normal(0, daily_range, n))
    open_p = low + np.random.random(n) * (high - low)

    # 成交量（对数正态，均值约 1.5亿）
    volume = np.random.lognormal(mean=18.8, sigma=0.5, size=n).astype(int)

    df = pd.DataFrame({
        "open": open_p, "high": high, "low": low, "close": close,
        "volume": volume,
        "turnover": volume * close * 0.8,
        "amplitude": (high - low) / close * 100,
        "pct_change": pd.Series(close).pct_change() * 100,
    }, index=dates)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    """Pickle 缓存"""
    with open(path, "wb") as f:
        pickle.dump(df, f)


def _load_cache(path: Path) -> pd.DataFrame | None:
    """读取 Pickle 缓存"""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, FileNotFoundError):
        return None


def _filter_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """按日期范围过滤"""
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


# ── 快捷入口 ──────────────────────────────────────

if __name__ == "__main__":
    df = fetch_boc_data("2020-01-01", "2025-06-30")
    print(df.head(10))
    print(f"\n总交易日: {len(df)}")
    print(f"\n统计摘要:\n{df.describe()}")
