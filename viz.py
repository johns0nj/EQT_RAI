import matplotlib.pyplot as plt
import numpy as np
from data_fetching import read_csv_file
import pandas as pd
from matplotlib.ticker import PercentFormatter, FuncFormatter  # 添加百分比格式化器和自定义格式化工具
import matplotlib.dates as mdates  # 导入日期格式化模块
from matplotlib.table import Table
from matplotlib.gridspec import GridSpec

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 清除缓存
plt.close('all')

# 自定义季度格式化函数
def format_quarter(x, pos=None):
    """
    格式化x轴日期为季度显示
    """
    try:
        date = mdates.num2date(x)
        return f'{date.year}-Q{(date.month - 1) // 3 + 1}'
    except Exception as e:
        print(f"日期格式化错误: {e}")
        return ''

def plot_rai(data=None):
    """
    绘制RAI数据的图表，并标注±1的位置
    """
    try:
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
        
        # 检查标普500数据是否存在
        if price_col not in rai_data.columns or rai_data[price_col].isna().all():
            print("警告：标普500数据为空，将仅绘制RAI图表")
            # 创建布局（仅RAI图表）
            fig = plt.figure(figsize=(18, 8))
            ax1 = plt.subplot(1, 1, 1)  # 仅一个子图
        else:
            # 创建布局（RAI和标普500图表）
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(3, 3, width_ratios=[3, 1, 1], height_ratios=[0.2, 1, 1])  # 3行3列，添加一个较小的顶部行
            ax1 = plt.subplot(gs[1, 0])  # RAI 图表
            ax2 = plt.subplot(gs[2, 0])  # 标普500图表
            ax_table = plt.subplot(gs[0:2, 1:])  # 使用右上方区域

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
        
        # 绘制RAI图表
        ax1.plot(rai_data.index, rai_data['World'], label='RAI', color='blue')
        
        # 添加±1的水平线
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='+100% 标准线')
        ax1.axhline(y=-100, color='green', linestyle='--', alpha=0.5, label='-100% 标准线')
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
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一次刻度
        ax1.xaxis.set_major_formatter(FuncFormatter(format_quarter))
        
        # 设置日期格式
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')  # 旋转标签并右对齐

        # 自动调整布局以防止标签被切off
        plt.gcf().autofmt_xdate(rotation=45, ha='right')

        # --- 绘制标普500图表（下方子图） ---
        if price_col in rai_data.columns and not rai_data[price_col].isna().all():
            # 绘制标普500价格
            ax2.plot(rai_data.index, rai_data[price_col], label='标普500价格', color='green')
            
            # 添加未来3个月下跌超过10%的阴影（与RAI图表同步）
            for start, end in crash_periods:
                ax2.axvspan(start, end, color='peachpuff', alpha=0.3, 
                           label='未来3个月内下跌>10%' if start == crash_periods[0][0] else "")
            
            # 获取最新价格
            latest_price = rai_data[price_col].iloc[-1]
            
            # 检查 latest_price 是否为 NaN
            if pd.isna(latest_price):
                # 如果最新价格为 NaN，则使用倒数第二个有效价格
                latest_price = rai_data[price_col].dropna().iloc[-1]
                print(f"警告：最新价格为 NaN，已使用倒数第二个有效价格：{latest_price}")

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
            
            # 设置 y 轴范围
            ax2.set_ylim(bottom=min(rai_data[price_col].min(), latest_price * 0.95),
                         top=max(rai_data[price_col].max(), latest_price * 1.05))
            
            # 设置标普500子图属性
            ax2.set_title('标普500价格走势')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('价格')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper left')

        # --- 计算胜率 ---
        if price_col in rai_data.columns and not rai_data[price_col].isna().all():
            # 计算1年（约252个交易日）的累计收益率
            holding_period = 252
            future_returns_1y = []
            for i in range(len(rai_data)):
                if i + holding_period < len(rai_data):
                    future_return = (rai_data[price_col].iloc[i+holding_period] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
                    future_returns_1y.append(future_return)
                else:
                    future_returns_1y.append(np.nan)
            
            rai_data['Future_Return_1Y'] = future_returns_1y
            
            # 找到RAI低于-100%的日期
            buy_signals = rai_data['World'] < -100
            
            # 计算胜率
            if buy_signals.sum() > 0:
                winning_trades = (rai_data.loc[buy_signals, 'Future_Return_1Y'] > 0.10).sum()
                total_trades = buy_signals.sum()
                win_rate = winning_trades / total_trades
            else:
                win_rate = 0.0
            
            # 找到RAI低于-150%的日期
            buy_signals_150 = rai_data['World'] < -150
            
            # 计算胜率
            if buy_signals_150.sum() > 0:
                winning_trades_150 = (rai_data.loc[buy_signals_150, 'Future_Return_1Y'] > 0.10).sum()
                total_trades_150 = buy_signals_150.sum()
                win_rate_150 = winning_trades_150 / total_trades_150
            else:
                win_rate_150 = 0.0
            
            # 找到RAI低于-200%的日期
            buy_signals_200 = rai_data['World'] < -200
            
            # 计算胜率
            if buy_signals_200.sum() > 0:
                winning_trades_200 = (rai_data.loc[buy_signals_200, 'Future_Return_1Y'] > 0.10).sum()
                total_trades_200 = buy_signals_200.sum()
                win_rate_200 = winning_trades_200 / total_trades_200
            else:
                win_rate_200 = 0.0
            
            # 计算年化交易次数
            total_years = (rai_data.index[-1] - rai_data.index[0]).days / 365.25
            trades_per_year = total_trades / total_years
            trades_per_year_150 = total_trades_150 / total_years
            trades_per_year_200 = total_trades_200 / total_years

            # 修改 calculate_max_drawdown 函数
            def calculate_max_drawdown(returns):
                """
                计算最大回撤
                :param returns: 收益率序列
                :return: 最大回撤（负值表示回撤）
                """
                # 打印调试信息
                print("\n计算最大回撤的收益率序列:")
                print(returns)
                
                if len(returns) == 0:
                    return 0.0
                
                # 计算累计收益率
                cumulative = (1 + returns).cumprod()
                
                # 计算峰值
                peak = cumulative.expanding(min_periods=1).max()
                
                # 计算回撤
                drawdown = (cumulative - peak) / peak
                
                # 返回最大回撤
                max_drawdown = drawdown.min()
                
                # 如果 max_drawdown 为 0 或接近 0，说明没有回撤
                if abs(max_drawdown) < 1e-10:  # 使用一个很小的阈值
                    return 0.0
                    
                return max_drawdown  # 返回负值表示回撤

            # 修改最大回撤计算逻辑
            if buy_signals_200.sum() > 0:
                # 打印调试信息
                print("\nRAI < -200% 的信号点数量:", buy_signals_200.sum())
                print("\n收益率序列:")
                print(rai_data.loc[buy_signals_200, 'Future_Return_1Y'])
                
                # 确保收益率序列不为空且不全为 NaN
                returns_200 = rai_data.loc[buy_signals_200, 'Future_Return_1Y'].dropna()
                if len(returns_200) > 0:
                    drawdown_200 = calculate_max_drawdown(returns_200)
                else:
                    print("\n警告：RAI < -200% 的收益率序列为空或全为 NaN")
                    drawdown_200 = 0.0
            else:
                drawdown_200 = 0.0

            # 创建合并表格数据
            table_data = [
                ['买入信号', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
                ['持有期', '1年', '1年', '1年'],
                ['目标收益', '>10%', '>10%', '>10%'],
                ['总交易次数', f'{total_trades}', f'{total_trades_150}', f'{total_trades_200}'],
                ['平均每年交易次数', f'{trades_per_year:.1f}', f'{trades_per_year_150:.1f}', f'{trades_per_year_200:.1f}'],
                ['获胜次数', f'{winning_trades}', f'{winning_trades_150}', f'{winning_trades_200}'],
                ['胜率', f'{win_rate:.1%}', f'{win_rate_150:.1%}', f'{win_rate_200:.1%}'],
                ['最大回撤', f'{calculate_max_drawdown(rai_data.loc[buy_signals, "Future_Return_1Y"]):.1%}', f'{calculate_max_drawdown(rai_data.loc[buy_signals_150, "Future_Return_1Y"]):.1%}', f'{drawdown_200:.1%}'],
            ]
            
            # 在图表下方添加表格
            ax_table.axis('off')
            
            # 创建表格
            table = ax_table.table(
                cellText=table_data,
                colLabels=['指标', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
                loc='center',
                cellLoc='center'
            )
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            
            # 设置标题
            ax_table.set_title('不同 RAI 水平下买入标普500并持有1年的表现', y=1.05, fontsize=14)

            # --- 创建 RAI < -200% 期间的收益表格 ---
            if buy_signals_200.sum() > 0:
                # 找到 RAI < -200% 的事件
                rai_below_200 = rai_data['World'] < -200
                events = []
                start_date = None
                
                # 遍历数据，找到 RAI < -200% 的区间
                for date, value in rai_below_200.items():
                    if value and start_date is None:
                        start_date = date  # 记录 RAI < -200% 的开始日期
                    elif not value and start_date is not None:
                        end_date = date  # 记录 RAI > -200% 的结束日期
                        events.append((start_date, end_date))
                        start_date = None
                
                # 计算每个事件期间的标普500收益率
                event_returns = []
                for start, end in events:
                    if end in rai_data.index:
                        start_price = rai_data.loc[start, price_col]
                        end_price = rai_data.loc[end, price_col]
                        return_pct = (end_price - start_price) / start_price
                        event_returns.append([start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), f'{return_pct:.1%}'])
                
                # 创建详细表格数据，并为收益添加渐变颜色
                event_table_data = event_returns
                
                # 创建新的子图用于显示详细表格
                ax_event_table = plt.subplot(gs[2, 1:])
                ax_event_table.axis('off')
                
                # 添加表格标题
                ax_event_table.set_title('RAI < -200% 期间的标普500表现', y=1.05, fontsize=14)
                
                # 创建详细表格
                event_table = ax_event_table.table(
                    cellText=event_table_data,
                    colLabels=['开始日期', '结束日期', '持有收益'],
                    loc='center',
                    cellLoc='center'
                )
                
                # 设置表格样式
                event_table.auto_set_font_size(False)
                event_table.set_fontsize(12)
                event_table.scale(1, 2.0)
                
                # 为收益添加渐变颜色
                returns_values = []
                for i in range(len(event_returns)):
                    return_str = event_returns[i][2].rstrip('%')
                    return_val = float(return_str) / 100
                    returns_values.append(return_val)
                
                # 计算最大和最小值用于归一化
                max_return = max(returns_values)
                min_return = min(returns_values)
                
                for i, return_val in enumerate(returns_values):
                    if return_val >= 0:
                        # 正收益用绿色渐变
                        intensity = return_val / max(max_return, 0.0001)  # 避免除以0
                        color = (0, min(0.8 * intensity + 0.2, 1), 0)  # RGB格式，绿色渐变
                    else:
                        # 负收益用红色渐变
                        intensity = return_val / min(min_return, -0.0001)  # 避免除以0
                        color = (min(0.8 * intensity + 0.2, 1), 0, 0)  # RGB格式，红色渐变
                    
                    event_table[(i+1, 2)].set_text_props(color=color)

    except Exception as e:
        print(f"发生错误: {e}")

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # 调整子图之间的间距

    plt.show()

if __name__ == "__main__":
    plot_rai()
