import matplotlib.pyplot as plt
import numpy as np
from data_fetching import read_csv_file
import pandas as pd
from matplotlib.ticker import PercentFormatter, FuncFormatter  # 添加百分比格式化器和自定义格式化工具
import matplotlib.dates as mdates  # 导入日期格式化模块

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 清除缓存
plt.close('all')

# 自定义季度格式化函数
def format_quarter(date_num, pos):
    date = mdates.num2date(date_num)  # 将数值日期转换为 datetime 对象
    year = date.year
    quarter = (date.month - 1) // 3 + 1
    return f'{year}-Q{quarter}'

def plot_rai(data=None):
    """
    绘制RAI数据的图表，并标注±1的位置
    """
    # 读取RAI.csv数据并获取标普价格列名
    rai_data, price_col = read_csv_file('RAI.csv')  # 获取price_col
    
    if rai_data is None:
        print("无法获取数据进行可视化")
        return
    
    # 数据预处理
    # 确保 Date 列是字符串类型
    rai_data['Date'] = rai_data['Date'].astype(str)
    
    # 删除包含 "Source" 或 "Produced" 的行
    rai_data = rai_data[~rai_data['Date'].str.contains('Source|Produced', na=False)]
    
    # 将World列转换为数值类型（在转换日期之前）
    rai_data['World'] = pd.to_numeric(rai_data['World'], errors='coerce')
    
    # 打印负值数量
    print(f"负值数量：{(rai_data['World'] < 0).sum()}")
    if (rai_data['World'] < 0).sum() > 0:
        print("负值示例：")
        print(rai_data[rai_data['World'] < 0].head())
    
    # 将数值转换为百分比（乘以100）
    rai_data['World'] = rai_data['World'] * 100
    
    # 添加更详细的数据检查
    print("\nRAI数据统计：")
    print(rai_data['World'].describe())
    print("\n数据类型：", rai_data['World'].dtype)
    print("\n是否有空值：", rai_data['World'].isna().sum())
    
    # 将日期转换为datetime格式，使用正确的格式
    rai_data['Date'] = pd.to_datetime(rai_data['Date'], format='%Y-%m-%d')
    rai_data.set_index('Date', inplace=True)
    
    # 创建双图布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 初始化 crash_periods
    crash_periods = []
    
    # --- 计算标普500未来3个月下跌超过10%的区间 ---
    if price_col in rai_data.columns:
        # 计算未来3个月（约63个交易日）的累计收益率
        future_window = 63  # 3个月的交易日数量
        future_returns = []
        for i in range(len(rai_data)):
            if i + future_window < len(rai_data):
                # 计算从当前日期到未来63个交易日的累计收益率
                future_return = (rai_data[price_col].iloc[i+future_window] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
                future_returns.append(future_return)
            else:
                future_returns.append(np.nan)
        
        rai_data['Future_Return'] = future_returns
        
        # 找到未来3个月累计下跌超过10%的日期
        crash_start = rai_data['Future_Return'] < -0.10
        
        # 找到对应的结束日期（即实际下跌超过10%的日期）
        for i in range(len(rai_data)):
            if crash_start.iloc[i]:
                start_date = rai_data.index[i]
                # 找到未来63个交易日内第一个累计下跌超过10%的日期
                for j in range(i+1, min(i+future_window+1, len(rai_data))):
                    if (rai_data[price_col].iloc[j] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i] < -0.10:
                        end_date = rai_data.index[j]
                        crash_periods.append((start_date, end_date))
                        break
    
    # --- 绘制RAI图表（上方子图） ---
    ax1.plot(rai_data.index, rai_data['World'], label='RAI', color='blue')
    
    # 添加标普500未来3个月下跌超过10%的阴影
    for start, end in crash_periods:
        ax1.axvspan(start, end, color='peachpuff', alpha=0.3, 
                   label='未来3个月内标普500下跌>10%' if start == crash_periods[0][0] else "")
    
    # 获取最新值并在图上标注
    latest_date = rai_data.index[-1]
    latest_value = rai_data['World'].iloc[-1]
    ax1.annotate(f'{latest_date.strftime("%Y-%m-%d")}\n{latest_value:.1f}%',
                xy=(latest_date, latest_value),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 添加水平虚线
    ax1.axhline(y=100, color='r', linestyle='--', label='±100% 标准线')
    ax1.axhline(y=-100, color='r', linestyle='--')
    ax1.axhline(y=-200, color='gray', linestyle='--', alpha=0.5, label='-200% 标准线')  # 修改为浅灰色，透明度0.5
    
    # 设置RAI子图属性
    ax1.set_title('Risk Appetite Indicator (RAI) 时间序列图')
    plt.draw()  # 强制刷新图表
    ax1.set_ylabel('RAI 值 (%)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')
    
    # 设置y轴范围，固定最大值为300%，最小值保持动态
    min_value = min(rai_data['World'].min(), -100)  # 确保至少包含 -100%
    max_value = 300  # 固定最大值为300%
    ax1.set_ylim(min_value - 10, max_value + 10)
    ax1.yaxis.set_major_formatter(PercentFormatter())

    # 设置x轴显示季度
    ax1.xaxis.set_major_formatter(FuncFormatter(format_quarter))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一次刻度

    # 自动旋转日期标签
    plt.gcf().autofmt_xdate(rotation=45)  # 旋转45度

    # 调整图表布局
    plt.tight_layout()

    # --- 绘制标普500图表（下方子图） ---
    if price_col in rai_data.columns:
        # 绘制标普500价格
        ax2.plot(rai_data.index, rai_data[price_col], label='标普500价格', color='green')
        
        # 添加未来3个月下跌超过10%的阴影（与RAI图表同步）
        for start, end in crash_periods:
            ax2.axvspan(start, end, color='peachpuff', alpha=0.3, 
                       label='未来3个月内下跌>10%' if start == crash_periods[0][0] else "")
        
        # 获取最新价格和日期
        latest_price = rai_data[price_col].iloc[-1]
        latest_date = rai_data.index[-1]
        
        # 在图上标注最新价格和日期
        ax2.annotate(f'{latest_date.strftime("%Y-%m-%d")}\n{latest_price:.2f}',
                    xy=(latest_date, latest_price),
                    xytext=(10, -20),  # 向右下方偏移
                    textcoords='offset points',
                    ha='left',  # 左对齐
                    va='top',   # 顶部对齐
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # 在标注之前打印一下最新值，用于调试
        print(f"最新日期: {latest_date}")
        print(f"最新价格: {latest_price}")
        
        # 调整y轴范围，确保最新价格可见
        ax2.set_ylim(bottom=min(rai_data[price_col].min(), latest_price * 0.95),
                     top=7000)
        
        # 设置标普500子图属性
        ax2.set_title('标普500价格走势')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('价格')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper left')

    plt.show()

if __name__ == "__main__":
    plot_rai()
