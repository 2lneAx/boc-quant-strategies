# 拉取历史数据（中国银行 601988，2010至今）
python main.py --fetch

# 列出所有策略
python main.py --list

# 运行放量突破策略 + 绘图
python main.py --strategy volume --plot

# 运行 PB 估值策略
python main.py --strategy pb --plot

# 全部策略 PK
python main.py --all

# 自定义参数
python main.py --strategy volume --params "price_period=20,vol_multiplier=1.5"

# 导出信号
python main.py --strategy volume --export signals.csv

# 策略诊断
python optimize.py --diagnose pb

# 参数优化（网格搜索）
python optimize.py --optimize volume
python optimize.py --optimize pb
python optimize.py --all
