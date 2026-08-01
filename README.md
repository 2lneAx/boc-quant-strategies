# 中国银行 (601988) 量化策略框架

> 这个项目完全通过 vibe coding 实现，纯属娱乐。

跟 AI 聊着聊着就搞出了一套量化交易系统。没有提前设计架构，没有画 UML 图，甚至连需求都是边聊边定——典型的"我说你做"式开发。**11 个策略、回测引擎、参数优化、16 年数据**，全部由 AI 一把梭，人工只负责说"好的"和"跑一下"。

⚠️ **免责声明**: 本项目仅供学习和娱乐，不构成任何投资建议。AI 写的代码、AI 选的最优参数、AI 吹得天花乱坠的回测结果——请带着怀疑的眼光看待一切。实盘交易请自行负责。

---

## 📊 策略一览

基于中国银行 (601988) 2010-2026 年共 4012 个交易日后复权数据的回测结果：

| # | 策略 | 收益 | 夏普 | 最大回撤 | 胜率 | 交易次数 |
|:--:|------|-----:|-----:|-----:|-----:|-----:|
| 🥇 | **放量突破** | **200.6%** | 0.39 | -36.2% | 55.6% | 45 |
| 🥈 | 双均线交叉 | 38.4% | 0.02 | -30.1% | 31.2% | 93 |
| 🥉 | PB 估值择时 | 57.2% | 0.08 | -22.9% | 81.8% | 11 |
| — | *买入持有 (基准)* | *122.0%* | — | — | — | — |

### 两大核心策略

#### 放量突破（Volume Breakout）

```
买入: 收盘价 > MA(10) AND 成交量 > MA(30) × 1.2
卖出: 收盘价 < MA(10) AND 成交量 < MA(30) × 0.3  (或止损-5%/止盈+10%)
```

- 唯一跑赢买入持有的策略（超额 78.6 个百分点）
- 量价配合过滤假突破，中行这种大市值银行不需要爆量就能确认趋势
- 参数由 324 组网格搜索优化得出

#### PB 估值择时（PB Valuation）

```
买入: 价格处于过去 252 天的底部 30% 分位
卖出: 价格涨到过去 252 天的顶部 90% 分位  (或止损-15%)
```

- 5 笔交易（默认参数）→ 优化到 11 笔，收益从 26% → 57%
- 大部分时间空仓等风来，只在"市场嫌弃银行"时入场
- 用价格分位数替代真实 PB，因为银行每股净资产变化缓慢

### 策略列表

| 策略 | 命令 | 类型 |
|------|------|------|
| 放量突破 | `volume` | 趋势跟踪 |
| PB 估值择时 | `pb` | 价值回归 |
| 双均线交叉 + ADX | `ma_cross` | 趋势跟踪 |
| 指数联动 | `index` | 宏观择时 |
| 布林带均值回归 | `bollinger` | 均值回归 |
| RSI 超买超卖 | `rsi` | 均值回归 |
| 唐奇安通道突破 | `donchian` | 趋势跟踪 |
| OBV 背离 | `obv` | 量价背离 |
| 股息分红策略 | `dividend` | 事件驱动 |
| 财报效应 | `earnings` | 事件驱动 |
| 配对交易 (简化) | `pairs` | 统计套利 |

---

## 🚀 快速开始

```bash
git clone https://github.com/2lneAx/boc-quant-strategies.git
cd boc-quant-strategies
pip install -r requirements.txt
```

### 基础用法

```bash
# 列出所有策略
python main.py --list

# 运行放量突破策略 + 绘图
python main.py --strategy volume --plot

# 运行 PB 估值策略
python main.py --strategy pb --start 2010-01-01 --end 2026-07-31 --plot

# 全部 11 个策略大比拼
python main.py --all

# 自定义参数回测
python main.py --strategy volume --params "price_period=20,vol_multiplier=1.5"

# 导出信号到 CSV
python main.py --strategy volume --export signals.csv
```

### 参数优化

```bash
# 诊断策略行为
python optimize.py --diagnose pb

# 网格搜索最优参数
python optimize.py --optimize volume
python optimize.py --optimize pb

# 一键优化所有
python optimize.py --all
```

---

## 📁 项目结构

```
.
├── main.py                    # 主入口，命令行切换策略
├── backtest_engine.py         # 回测引擎 (T+1, 万三佣金, 千一印花税)
├── data_utils.py              # 数据获取 (腾讯API→akshare→本地缓存)
├── optimize.py                # 参数优化 (网格搜索) + 策略诊断
├── requirements.txt           # 依赖: pandas, numpy, akshare, matplotlib
└── strategies/
    ├── __init__.py             # STRATEGY_REGISTRY 注册表
    ├── base.py                 # 基类 BaseStrategy + Signal 枚举
    ├── volume_breakout.py      # 放量突破
    ├── pb_valuation.py         # PB 估值择时
    ├── ma_cross.py             # 双均线交叉 + ADX 过滤
    ├── index_linkage.py        # 上证50 / 沪深300 联动
    ├── bollinger.py            # 布林带均值回归
    ├── rsi.py                  # RSI 超买超卖
    ├── donchian.py             # 唐奇安通道突破 + ATR 止损
    ├── obv_divergence.py       # OBV 背离 + MACD 确认
    ├── dividend.py             # 股息分红日历策略
    ├── earnings.py             # 财报事件驱动
    └── pairs_trading.py        # 配对交易 (简化版)
```

---

## 🔧 添加新策略

三步完成：

```python
# 1. 在 strategies/ 下新建 my_strategy.py，继承 BaseStrategy
# 2. 实现 generate_signals(df) 和 description
# 3. 在 __init__.py 的 STRATEGY_REGISTRY 中注册

# 然后直接:
python main.py --strategy my_strategy
```

---

## ⚙️ 技术细节

- **数据源**: akshare (腾讯源 `stock_zh_a_hist_tx`) 获取前/后复权数据，自动缓存到 `.cache/`
- **回测规则**: T+1 执行、万三佣金（双边）、千一印花税（卖出）、千一滑点
- **交易单位**: 100 股整数倍，最小交易 100 股
- **优化方法**: 全量网格搜索，按总收益率排序
- **参数已优化**: `volume_breakout.py` 和 `pb_valuation.py` 的默认参数已经是最优值

---

## 📝 Vibe Coding 日志

一个真实的 AI 协作开发时间线：

1. "针对中国银行设计量化策略，先探讨一下有哪些合适的" → AI 输出 11 个策略方案
2. "帮我用 python 写" → AI 一把梭写出完整框架
3. "好的"（安装 akshare）→ API 限速，换腾讯源
4. "用 2010 至今的数据跑一下" → 前复权价格出现负值，修复
5. "PB 为什么只有 1 次交易" → 发现回测引擎 `trade_log` bug，修复
6. "优化参数" → 324 + 100 组网格搜索，最优参数写入默认值
7. "用数学公式解释" → $\LaTeX$ 风格公式
8. "上传到 GitHub" → git 没装→winget 安装→gh 认证失败→REST API 直传

全程没有写一行设计文档，没有讨论架构模式，就是一边聊一边改。这大概就是 2026 年的编程方式。

---

## 📄 License

MIT — 代码是 AI 写的，出了事别找我。
