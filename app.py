import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_fetching import read_csv_file
import numpy as np

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
        total_trades = buy_signals.sum()
        
        # 计算胜率
        if total_trades > 0:
            winning_trades = (rai_data.loc[buy_signals, 'Future_Return_1Y'] > 0.10).sum()
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0.0
        
        # 找到RAI低于-150%的日期
        buy_signals_150 = rai_data['World'] < -150
        total_trades_150 = buy_signals_150.sum()
        
        # 计算胜率
        if total_trades_150 > 0:
            winning_trades_150 = (rai_data.loc[buy_signals_150, 'Future_Return_1Y'] > 0.10).sum()
            win_rate_150 = winning_trades_150 / total_trades_150
        else:
            win_rate_150 = 0.0
        
        # 找到RAI低于-200%的日期
        buy_signals_200 = rai_data['World'] < -200
        total_trades_200 = buy_signals_200.sum()
        
        # 计算胜率
        if total_trades_200 > 0:
            winning_trades_200 = (rai_data.loc[buy_signals_200, 'Future_Return_1Y'] > 0.10).sum()
            win_rate_200 = winning_trades_200 / total_trades_200
        else:
            win_rate_200 = 0.0
        
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
        max_drawdown = calculate_max_drawdown(rai_data.loc[buy_signals, 'Future_Return_1Y'])
        max_drawdown_150 = calculate_max_drawdown(rai_data.loc[buy_signals_150, 'Future_Return_1Y'])
        max_drawdown_200 = calculate_max_drawdown(rai_data.loc[buy_signals_200, 'Future_Return_1Y'])

    # 创建第一个表格数据
    table_data = [
        ['买入信号', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
        ['持有期', '1年', '1年', '1年'],
        ['目标收益', '>10%', '>10%', '>10%'],
        ['总交易次数', f'{total_trades}', f'{total_trades_150}', f'{total_trades_200}'],
        ['平均每年交易次数', f'{trades_per_year:.1f}', f'{trades_per_year_150:.1f}', f'{trades_per_year_200:.1f}'],
        ['获胜次数', f'{winning_trades}', f'{winning_trades_150}', f'{winning_trades_200}'],
        ['胜率', f'{win_rate:.1%}', f'{win_rate_150:.1%}', f'{win_rate_200:.1%}'],
        ['最大回撤', f'{max_drawdown:.1%}', f'{max_drawdown_150:.1%}', f'{max_drawdown_200:.1%}'],
    ]

    # 创建第一个表格
    table1 = go.Table(
        header=dict(values=['指标', 'RAI < -100%', 'RAI < -150%', 'RAI < -200%'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=list(zip(*table_data)),
                   fill_color='lavender',
                   align='left')
    )

    # 创建第二个表格数据
    event_table_data = [
        ['2020-03-01', '2020-03-23', '-20.5%'],
        ['2022-01-03', '2022-01-27', '-12.3%'],
        # 添加更多事件数据...
    ]

    # 创建第二个表格
    table2 = go.Table(
        header=dict(values=['开始日期', '结束日期', '持有收益'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=list(zip(*event_table_data)),
                   fill_color='lavender',
                   align='left')
    )

    # 创建子图布局
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Appetite Indicator (RAI) 时间序列图', 
                       '标普500价格走势',
                       'RAI 分析结果',
                       'RAI < -200% 后标普500表现'),  # 添加第4个子图标题
        vertical_spacing=0.1,
        horizontal_spacing=0.2,
        row_heights=[0.5, 0.5],
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter", "colspan": 1}, {"type": "table"}],
               [{"type": "scatter", "colspan": 1}, {"type": "table"}]]  # 为 (2,2) 分配表格子图
    )

    # 添加图表和表格
    fig.add_trace(
        go.Scatter(
            x=rai_data.index,
            y=rai_data['World'],
            name='RAI',
            line=dict(color='blue')
        ),
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
                line=dict(color='green')
            ),
            row=2, col=1
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

    # 找到 RAI 第一次小于 -200% 的日期
    first_below_200 = rai_data[rai_data['World'] < -200].index.min()
    
    # 获取之后1年内的标普500价格数据
    if first_below_200 and price_col in rai_data.columns:
        one_year_later = first_below_200 + pd.Timedelta(days=365)
        price_data = rai_data.loc[first_below_200:one_year_later, price_col]
        
        # 计算每日价格变动
        daily_returns = price_data.pct_change().dropna()
        
        # 创建表格数据
        table_data = [
            ['日期', '价格', '日收益率'],
            *[[date.strftime('%Y-%m-%d'), f'{price:.2f}', f'{ret:.2%}'] 
              for date, price, ret in zip(price_data.index, price_data, daily_returns)]
        ]
        
        # 创建表格
        table2 = go.Table(
            header=dict(values=['日期', '价格', '日收益率'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=list(zip(*table_data)),
                      fill_color='lavender',
                      align='left')
        )
        
        # 将表格添加到右下角
        fig.add_trace(table2, row=2, col=2)

    # 更新布局
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="RAI 和标普500分析图表",
        title_x=0.5,
        title_font_size=20,
    )
    
    # 更新 RAI 图表的 Y 轴范围
    min_value = min(rai_data['World'].min(), -100)
    fig.update_yaxes(
        range=[min_value - 10, 300],
        title_text="RAI 值 (%)",
        row=1, col=1,
        gridcolor='lightgray',
        gridwidth=0.5,
    )
    
    # 更新标普500图表的 Y 轴范围
    if price_col in rai_data.columns:
        # 获取最新价格
        latest_price = rai_data[price_col].iloc[-1]
        
        # 计算y轴范围，确保最新价格完全可见
        y_min = min(rai_data[price_col].min(), latest_price * 0.95)
        y_max = max(rai_data[price_col].max(), latest_price * 1.05)
        
        fig.update_yaxes(
            range=[y_min, y_max],
            title_text="价格",
            row=2, col=1,
            gridcolor='lightgray',
            gridwidth=0.5,
        )
    
    # 更新 X 轴范围，增加1个月留白
    x_min = rai_data.index.min()
    x_max = rai_data.index.max() + pd.Timedelta(days=30)  # 增加1个月留白
    
    fig.update_xaxes(
        range=[x_min, x_max],
        title_text="时间",
        row=2, col=1,
        gridcolor='lightgray',
        gridwidth=0.5,
    )
    
    # 更新RAI图表的X轴范围
    fig.update_xaxes(
        range=[x_min, x_max],
        row=1, col=1,
        gridcolor='lightgray',
        gridwidth=0.5,
    )
    
    return fig

# 设置布局
app.layout = html.Div([
    dcc.Graph(id='main-figure', figure=create_figures(), style={'height': '800px'})
])

if __name__ == '__main__':
    print("Dash 应用正在运行，请访问：http://127.0.0.1:8050/")
    app.run_server(debug=True)
