import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_fetching import read_csv_file
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter, FuncFormatter
import matplotlib.dates as mdates

# 初始化 Dash 应用
app = dash.Dash(__name__)

def create_figures():
    # 读取数据
    rai_data, price_col = read_csv_file('RAI.csv')
    
    if rai_data is None:
        return go.Figure()
    
    # 数据预处理
    rai_data['Date'] = pd.to_datetime(rai_data['Date'])
    rai_data.set_index('Date', inplace=True)
    rai_data['World'] = pd.to_numeric(rai_data['World'], errors='coerce') * 100
    
    # 计算标普500未来3个月下跌超过10%的区间
    crash_periods = []
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
    
    # 计算交易次数和胜率
    if price_col in rai_data.columns:
        # 计算半年（约126个交易日）的累计收益率
        half_year_period = 126
        future_returns_6m = []
        for i in range(len(rai_data)):
            if i + half_year_period < len(rai_data):
                future_return = (rai_data[price_col].iloc[i+half_year_period] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
                future_returns_6m.append(future_return)
            else:
                future_returns_6m.append(np.nan)
        
        rai_data['Future_Return_6M'] = future_returns_6m
        
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
        
        # 计算不同RAI水平下的胜率
        def calculate_win_rates(signals):
            if signals.sum() > 0:
                win_rate_6m_10 = (rai_data.loc[signals, 'Future_Return_6M'] > 0.10).sum() / signals.sum()
                win_rate_1y_20 = (rai_data.loc[signals, 'Future_Return_1Y'] > 0.20).sum() / signals.sum()
                win_rate_1y_10 = (rai_data.loc[signals, 'Future_Return_1Y'] > 0.10).sum() / signals.sum()
            else:
                win_rate_6m_10 = win_rate_1y_20 = win_rate_1y_10 = 0.0
            return win_rate_6m_10, win_rate_1y_20, win_rate_1y_10

        # RAI < -100%
        buy_signals = rai_data['World'] < -100
        win_rate_6m_10_100, win_rate_1y_20_100, win_rate_1y_10_100 = calculate_win_rates(buy_signals)
        
        # RAI < -150%
        buy_signals_150 = rai_data['World'] < -150
        win_rate_6m_10_150, win_rate_1y_20_150, win_rate_1y_10_150 = calculate_win_rates(buy_signals_150)
        
        # RAI < -200%
        buy_signals_200 = rai_data['World'] < -200
        win_rate_6m_10_200, win_rate_1y_20_200, win_rate_1y_10_200 = calculate_win_rates(buy_signals_200)
        
        # 找到RAI低于-100%的日期
        buy_signals_100 = rai_data['World'] < -100
        total_trades = buy_signals_100.sum()
        
        # 找到RAI低于-150%的日期
        buy_signals_150 = rai_data['World'] < -150
        total_trades_150 = buy_signals_150.sum()
        
        # 找到RAI低于-200%的日期
        buy_signals_200 = rai_data['World'] < -200
        total_trades_200 = buy_signals_200.sum()
        
        # 计算年化交易次数
        total_years = (rai_data.index[-1] - rai_data.index[0]).days / 365.25
        trades_per_year = total_trades / total_years
        trades_per_year_150 = total_trades_150 / total_years
        trades_per_year_200 = total_trades_200 / total_years

        # 计算最大回撤
        def calculate_max_drawdown(returns):
            """
            计算最大回撤
            :param returns: 收益率序列
            :return: 最大回撤（负值表示回撤）
            """
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

        # 计算最大回撤
        max_drawdown = calculate_max_drawdown(rai_data.loc[buy_signals_100, 'Future_Return_1Y'])
        max_drawdown_150 = calculate_max_drawdown(rai_data.loc[buy_signals_150, 'Future_Return_1Y'])
        max_drawdown_200 = calculate_max_drawdown(rai_data.loc[buy_signals_200, 'Future_Return_1Y'])

    # 创建表格数据
    table_data = [
        ['买入信号', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
        ['持有期', '1年', '1年', '1年'],
        ['目标收益', '>10%', '>10%', '>10%'],
        ['总交易次数', f'{total_trades}', f'{total_trades_150}', f'{total_trades_200}'],
        ['平均每年交易次数', f'{trades_per_year:.1f}', f'{trades_per_year_150:.1f}', f'{trades_per_year_200:.1f}'],
        ['获胜次数', f'{total_trades}', f'{total_trades_150}', f'{total_trades_200}'],
        ['胜率(1年>10%)', f'{win_rate_1y_10_100:.1%}', f'{win_rate_1y_10_150:.1%}', f'{win_rate_1y_10_200:.1%}'],
        ['胜率(半年>10%)', f'{win_rate_6m_10_100:.1%}', f'{win_rate_6m_10_150:.1%}', f'{win_rate_6m_10_200:.1%}'],
        ['胜率(1年>20%)', f'{win_rate_1y_20_100:.1%}', f'{win_rate_1y_20_150:.1%}', f'{win_rate_1y_20_200:.1%}'],
        ['最大回撤', f'{max_drawdown:.1%}', f'{max_drawdown_150:.1%}', f'{max_drawdown_200:.1%}'],
    ]

    # 定义颜色映射函数
    def get_color(value):
        if value >= 0:
            # 绿色，值越大颜色越深
            intensity = min(int(value * 255), 255)
            return f'rgb(0, {intensity}, 0)'
        else:
            # 红色，值越小颜色越深
            intensity = min(int(abs(value) * 255), 255)
            return f'rgb({intensity}, 0, 0)'

    # 创建表格
    table1 = go.Table(
        header=dict(
            values=['指标', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14),
            height=45
        ),
        cells=dict(
            values=list(zip(*table_data)),
            fill_color='lavender',
            align='center',
            font=dict(size=14),
            height=40,
            # 为胜率行设置颜色
            fill=dict(
                color=[
                    ['white'] * 4,  # 第一行
                    ['white'] * 4,  # 第二行
                    ['white'] * 4,  # 第三行
                    ['white'] * 4,  # 第四行
                    ['white'] * 4,  # 第五行
                    ['white'] * 4,  # 第六行
                    [get_color(win_rate_1y_10_100), get_color(win_rate_1y_10_150), get_color(win_rate_1y_10_200)],  # 胜率(1年>10%)
                    [get_color(win_rate_6m_10_100), get_color(win_rate_6m_10_150), get_color(win_rate_6m_10_200)],  # 胜率(半年>10%)
                    [get_color(win_rate_1y_20_100), get_color(win_rate_1y_20_150), get_color(win_rate_1y_20_200)],  # 胜率(1年>20%)
                    ['white'] * 4  # 最大回撤
                ]
            )
        ),
        columnwidth=[400, 200, 200, 200],
    )

    # 创建子图布局
    fig = make_subplots(
        rows=2, cols=2,  # 2行2列
        subplot_titles=(
            'Risk Appetite Indicator (RAI) 时间序列图', 
            '标普500价格走势',
            None,  # 空出右下角
            'RAI 分析结果'
        ),
        vertical_spacing=0.2,  # 增加垂直间距
        horizontal_spacing=0.3,  # 增加水平间距
        row_heights=[0.5, 0.5],  # 调整行高比例
        column_widths=[0.6, 0.4],  # 调整列宽比例
        specs=[
            [{"type": "scatter"}, {"type": "table", "rowspan": 2}],  # 第一列垂直排列图表
            [{"type": "scatter"}, None]  # 第二列表格跨两行
        ]
    )

    # 添加RAI图表
    fig.add_trace(
        go.Scatter(
            x=rai_data.index,
            y=rai_data['World'],
            name='RAI',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1  # 第一行第一列
    )
    
    # 添加最新RAI值和日期的标注
    latest_date = rai_data.index[-1]
    latest_rai = rai_data['World'].iloc[-1]
    
    fig.add_annotation(
        x=latest_date,
        y=latest_rai,
        text=f'最新日期: {latest_date.strftime("%Y-%m-%d")}<br>RAI: {latest_rai:.1f}%',
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-40,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='blue',
        borderwidth=2,
        borderpad=4,
        row=1, col=1
    )
    
    # 添加阴影区域到 RAI 图表
    for start, end in crash_periods:
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=min(rai_data['World'].min(), -100) - 10,
            y1=300,
            fillcolor="peachpuff",
            opacity=0.3,
            layer="below",
            line_width=0,
            row=1, col=1
        )
    
    # 添加水平参考线
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-200, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # 添加标普500图表
    if price_col in rai_data.columns:
        fig.add_trace(
            go.Scatter(
                x=rai_data.index,
                y=rai_data[price_col],
                name='标普500价格',
                line=dict(color='green', width=2)
            ),
            row=2, col=1  # 第二行第一列
        )
        
        # 添加阴影区域到标普500图表
        for start, end in crash_periods:
            fig.add_shape(
                type="rect",
                x0=start,
                x1=end,
                y0=rai_data[price_col].min() * 0.95,
                y1=7000,
                fillcolor="peachpuff",
                opacity=0.3,
                layer="below",
                line_width=0,
                row=2, col=1
            )
    
    # 将表格添加到右侧
    fig.add_trace(table1, row=1, col=2)

    # 更新布局
    fig.update_layout(
        height=1200,  # 调整总高度
        showlegend=True,
        title_text="RAI 和标普500分析图表",
        title_x=0.5,
        title_font_size=24,
        margin=dict(l=50, r=50, t=100, b=150),
        font=dict(size=12),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新坐标轴标签字体大小
    fig.update_xaxes(title_font=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14))

    # 更新子图标题字体大小
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=16))

    return fig

# 设置布局
app.layout = html.Div([
    dcc.Graph(id='main-figure', figure=create_figures(), style={'height': '800px'})
])

def format_quarter(x, pos=None):
    """
    格式化x轴日期为季度显示
    """
    return f'{x.year} Q{(x.month - 1) // 3 + 1}'

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

        # 初始化 crash_periods
        crash_periods = []
        
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
        ax1.xaxis.set_major_formatter(FuncFormatter(format_quarter))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月显示一次刻度

        # 自动旋转日期标签
        plt.gcf().autofmt_xdate(rotation=45)  # 旋转45度

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
            ax2.annotate(f'{latest_date.strftime("%Y-%m-%d")}\n{latest_price:.2f}', # type: ignore
                        xy=(latest_date, latest_price), # type: ignore
                        xytext=(10, -20),  # 向右下方偏移
                        textcoords='offset points',
                        ha='left',  # 左对齐
                        va='top',   # 顶部对齐
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # 在标注之前打印一下最新值，用于调试
            print(f"最新日期: {latest_date}") # type: ignore
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

    except Exception as e:
        print(f"发生错误: {e}")

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)  # 调整子图之间的间距

    plt.show()

if __name__ == '__main__':
    print("Dash 应用正在运行，请访问：http://127.0.0.1:8050/")
    app.run_server(debug=True)
