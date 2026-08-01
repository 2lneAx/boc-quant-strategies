"""
中国银行 (601988) 量化策略框架 — 主入口

用法:
  # 列出所有策略
  python main.py --list

  # 运行单个策略
  python main.py --strategy bollinger

  # 运行多个策略对比
  python main.py --strategy bollinger,rsi,ma_cross

  # 运行全部策略
  python main.py --all

  # 自定义参数
  python main.py --strategy bollinger --params "period=30,k=2.5"

  # 指定日期范围
  python main.py --strategy pb --start 2020-01-01 --end 2025-12-31

  # 绘图
  python main.py --strategy bollinger --plot

  # 导出信号到 CSV
  python main.py --strategy bollinger --export signals.csv
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import fetch_boc_data
from strategies import STRATEGY_REGISTRY, BaseStrategy
from strategies.combo import ComboStrategy, print_combo_result
from backtest_engine import BacktestEngine, BacktestResult, print_result


# ── 命令行参数 ──────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="中国银行 (601988) 量化策略回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --list                          # 列出所有策略
  python main.py --strategy bollinger             # 运行布林带策略
  python main.py --all                            # 运行全部策略对比
  python main.py --strategy rsi --plot            # 运行并绘图
  python main.py --strategy bollinger --params "period=30,k=2.5"  # 自定义参数
  python main.py --strategy pb --start 2020-01-01 --end 2025-06-30
        """,
    )
    p.add_argument("--list", action="store_true", help="列出所有可用策略及简介")
    p.add_argument("--strategy", "-s", type=str, help="运行策略名称，多个用逗号分隔")
    p.add_argument("--all", action="store_true", help="运行全部策略对比")
    p.add_argument("--params", type=str, help="策略参数 key=val,key=val 格式")
    p.add_argument("--start", type=str, default="2018-01-01", help="回测起始日期")
    p.add_argument("--end", type=str, default="2025-12-31", help="回测结束日期")
    p.add_argument("--plot", action="store_true", help="绘制权益曲线")
    p.add_argument("--export", type=str, help="导出信号到 CSV 文件")
    p.add_argument("--capital", type=float, default=100_000.0, help="初始资金")
    return p.parse_args()


# ── 主逻辑 ──────────────────────────────────────

def list_strategies():
    """列出所有已注册的策略"""
    print(f"\n{'='*65}")
    print(f"  中国银行 (601988) 量化策略库 — 共 {len(STRATEGY_REGISTRY)} 个策略")
    print(f"{'='*65}\n")
    for i, (name, cls) in enumerate(STRATEGY_REGISTRY.items(), 1):
        instance = cls()
        print(f"  [{i:2d}] {name:<15s}  {instance.description}")
    print(f"\n  用法: python main.py --strategy <name>")
    print(f"  示例: python main.py --strategy bollinger\n")


def parse_params(params_str: str) -> dict:
    """解析参数字符串 'key=val,key=val' → dict"""
    if not params_str:
        return {}
    result = {}
    for pair in params_str.split(","):
        k, v = pair.strip().split("=")
        # 自动类型转换
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        result[k] = v
    return result


def run_strategy(name: str, df, args) -> BacktestResult | None:
    """运行单个策略并返回回测结果"""
    if name not in STRATEGY_REGISTRY:
        print(f"[错误] 未知策略 '{name}'。可用策略: {list(STRATEGY_REGISTRY.keys())}")
        return None

    # 实例化策略
    strategy_cls = STRATEGY_REGISTRY[name]
    strategy = strategy_cls()

    # 应用自定义参数
    if args.params:
        try:
            custom_params = parse_params(args.params)
            strategy.set_params(**custom_params)
            print(f"[参数] 已应用自定义参数: {custom_params}")
        except Exception as e:
            print(f"[警告] 参数解析失败: {e}，使用默认参数")

    print(f"\n[策略] {strategy.description}")

    # 生成信号
    result = strategy.run(df)

    # 组合策略走独立回测逻辑（双账户分仓）
    if isinstance(strategy, ComboStrategy):
        result = strategy.run(df, initial_capital=args.capital)
        print_combo_result(result)

        if args.plot:
            plot_combo_result(result, strategy.description)

        return result

    # 标准回测
    engine = BacktestEngine(initial_capital=args.capital)
    bt_result = engine.run(df, result.signals)

    # 打印
    print_result(bt_result, strategy.description)

    # 导出
    if args.export:
        export_path = args.export
        export_df = df.copy()
        export_df["signal"] = result.signals
        export_df["position"] = result.positions
        export_df.to_csv(export_path, encoding="utf-8-sig")
        print(f"[导出] 信号已保存至: {export_path}")

    # 绘图
    if args.plot:
        plot_result(bt_result, strategy.description)

    return bt_result


def plot_combo_result(result, title: str):
    """绘制组合策略权益曲线"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        equity = result.equity_curve
        m = result.metrics

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        ax1 = axes[0]
        ax1.plot(equity.index, equity.values, label="组合", linewidth=1.5, color="#2ca02c")
        ax1.axhline(y=100_000, color="gray", linewidth=0.5, linestyle=":")
        ax1.set_title(f"组合策略 — {title}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("权益 (元)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        info_text = (f"收益: {m['total_return']:.1%}  "
                     f"夏普: {m['sharpe_ratio']:.2f}  "
                     f"回撤: {m['max_drawdown']:.1%}")
        ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax2 = axes[1]
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=0.8)
        ax2.set_ylabel("回撤 (%)")
        ax2.set_xlabel("日期")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[绘图] 绘图失败: {e}")


def plot_result(bt_result: BacktestResult, title: str):
    """绘制权益曲线对比"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        # 上图: 权益曲线
        ax1 = axes[0]
        ax1.plot(bt_result.equity_curve.index, bt_result.equity_curve.values,
                 label="策略", linewidth=1.5, color="#1f77b4")
        ax1.plot(bt_result.benchmark_curve.index, bt_result.benchmark_curve.values,
                 label="买入持有", linewidth=1, color="#ff7f0e", alpha=0.7, linestyle="--")
        ax1.axhline(y=100_000, color="gray", linewidth=0.5, linestyle=":")
        ax1.set_title(f"权益曲线 — {title}", fontsize=14, fontweight="bold")
        ax1.set_ylabel("权益 (元)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        # 下图: 回撤
        ax2 = axes[1]
        peak = bt_result.equity_curve.expanding().max()
        drawdown = (bt_result.equity_curve - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color="#d62728", alpha=0.3, label="回撤 %")
        ax2.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=0.8)
        ax2.set_ylabel("回撤 (%)")
        ax2.set_xlabel("日期")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[绘图] 绘图失败: {e}")


def run_all(df, args):
    """运行全部策略并排名"""
    results = []
    for name, cls in STRATEGY_REGISTRY.items():
        strategy = cls()

        # 组合策略特殊处理
        if isinstance(strategy, ComboStrategy):
            sr = strategy.run(df, initial_capital=args.capital)
            m = sr.metrics
            print(f"  ✓ {name:<15s}  {m['total_return']:>8.2%}  Sharpe={m['sharpe_ratio']:>6.2f}  "
                  f"MDD={m['max_drawdown']:>8.2%}  Vol={m['vol_trades']}+PB={m['pb_trades']}笔")
            results.append((name, strategy.description, sr))
            continue

        sr = strategy.run(df)
        engine = BacktestEngine(initial_capital=args.capital)
        bt = engine.run(df, sr.signals)
        results.append((name, strategy.description, bt))
        print(f"  ✓ {name:<15s}  {bt.total_return:>8.2%}  Sharpe={bt.sharpe_ratio:>6.2f}  "
              f"MDD={bt.max_drawdown:>8.2%}  WinRate={bt.win_rate:>6.1%}")

    # 排名
    print(f"\n{'='*60}")
    print(f"  策略排名（按总收益率）")
    print(f"{'='*60}")

    def _get_return(item):
        _, _, obj = item
        if hasattr(obj, "metrics"):
            return obj.metrics["total_return"]
        return obj.total_return

    sorted_results = sorted(results, key=_get_return, reverse=True)
    for rank, (name, desc, obj) in enumerate(sorted_results, 1):
        if hasattr(obj, "metrics"):
            ret = obj.metrics["total_return"]
            sharpe = obj.metrics["sharpe_ratio"]
            mdd = obj.metrics["max_drawdown"]
        else:
            ret = obj.total_return
            sharpe = obj.sharpe_ratio
            mdd = obj.max_drawdown
        print(f"  {rank:2d}. {name:<15s} {ret:>8.2%}  Sharpe={sharpe:.2f}  "
              f"MDD={mdd:.2%}  {desc}")


# ── 入口 ────────────────────────────────────────

def main():
    args = parse_args()

    # --list: 只列出策略
    if args.list:
        list_strategies()
        return

    # 加载数据
    print(f"[数据] 加载中国银行 (601988) 数据: {args.start} ~ {args.end}")
    df = fetch_boc_data(args.start, args.end)
    if df.empty:
        print("[错误] 数据为空，请检查数据源或日期范围")
        return
    print(f"[数据] 共 {len(df)} 个交易日")

    # 运行策略
    if args.all:
        print(f"\n[运行] 全部策略对比模式")
        print(f"{'─'*60}")
        run_all(df, args)
    elif args.strategy:
        names = [n.strip() for n in args.strategy.split(",")]
        for name in names:
            run_strategy(name, df, args)
    else:
        # 默认：交互模式
        list_strategies()
        print("请选择策略运行: python main.py --strategy <名称>")
        print("或运行全部对比: python main.py --all")


if __name__ == "__main__":
    main()
