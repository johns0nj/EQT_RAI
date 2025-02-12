import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_fetching import read_csv_file
import plotly.subplots as sp

# 设置中文字体
import plotly.io as pio
pio.templates["custom"] = go.layout.Template(
    layout=dict(
        font=dict(family="SimHei")  # 设置中文字体
    )
)
pio.templates.default = "custom"

# 读取数据
rai_data, price_col = read_csv_file('RAI.csv')

# 数据预处理
rai_data['Date'] = pd.to_datetime(rai_data['Date'], format='%Y-%m-%d')
rai_data.set_index('Date', inplace=True)
rai_data['World'] = pd.to_numeric(rai_data['World'], errors='coerce') * 100

# 计算未来1年收益率
holding_period = 252
future_returns_1y = []
for i in range(len(rai_data)):
    if i + holding_period < len(rai_data):
        future_return = (rai_data[price_col].iloc[i+holding_period] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
        future_returns_1y.append(future_return)
    else:
        future_returns_1y.append(np.nan)
rai_data['Future_Return_1Y'] = future_returns_1y

# 初始化 Dash 应用
app = dash.Dash(__name__)

# 布局
app.layout = html.Div([
    # 创建2行3列的子图布局
    dcc.Graph(id='main-figure', style={'height': '800px'}),
    html.Div(id='stats-table', style={'margin': '20px'}),
    html.Div(id='event-table', style={'margin': '20px'})
])

# 回调函数
@app.callback(
    [Output('main-figure', 'figure'),
     Output('stats-table', 'children'),
     Output('event-table', 'children')],
    [Input('main-figure', 'relayoutData')]
)
def update_charts(relayoutData):
    # 创建子图
    fig = sp.make_subplots(
        rows=2, cols=1,  # 改为2行1列
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1,  # 添加垂直间距
        subplot_titles=('Risk Appetite Indicator (RAI) 时间序列图', '标普500价格走势')
    )

    # 计算标普500未来3个月下跌超过10%的区间
    future_window = 63  # 3个月的交易日数量
    crash_periods = []
    for i in range(len(rai_data)):
        if i + future_window < len(rai_data):
            future_return = (rai_data[price_col].iloc[i+future_window] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
            if future_return < -0.10:
                start_date = rai_data.index[i]
                end_date = rai_data.index[i+future_window]
                crash_periods.append((start_date, end_date))

    # RAI 图表
    fig.add_trace(
        go.Scatter(x=rai_data.index, y=rai_data['World'], name='RAI', mode='lines'),  # 添加 mode='lines'
        row=1, col=1
    )

    # 添加水平线
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=1, col=1, annotation_text="100%")
    fig.add_hline(y=-100, line_dash="dash", line_color="red", row=1, col=1, annotation_text="-100%")
    fig.add_hline(y=-200, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1, annotation_text="-200%")

    # 添加阴影区域
    for start, end in crash_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="peachpuff", opacity=0.3,
            layer="below", line_width=0,
            row=1, col=1,
            annotation_text="未来3个月内标普500下跌>10%",
            annotation_position="top left"
        )

    # 标普500图表
    fig.add_trace(
        go.Scatter(x=rai_data.index, y=rai_data[price_col], name='标普500价格', mode='lines'),  # 添加 mode='lines'
        row=2, col=1
    )

    # 添加阴影区域到标普500图表
    for start, end in crash_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="peachpuff", opacity=0.3,
            layer="below", line_width=0,
            row=2, col=1,
            annotation_text="未来3个月内下跌>10%",
            annotation_position="top left"
        )

    # 更新布局
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="RAI 和标普500分析图表",
        margin=dict(t=100, b=50),  # 调整边距
        template="custom"  # 使用自定义模板
    )

    # 更新x轴和y轴
    fig.update_xaxes(title_text="时间", row=1, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="RAI 值 (%)", row=1, col=1)
    fig.update_yaxes(title_text="价格", row=2, col=1)

    # 计算胜率和最大回撤
    def calculate_max_drawdown(returns):
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    buy_signals = rai_data['World'] < -100
    buy_signals_150 = rai_data['World'] < -150
    buy_signals_200 = rai_data['World'] < -200

    # 计算年化交易次数
    total_years = (rai_data.index[-1] - rai_data.index[0]).days / 365.25
    total_trades = buy_signals.sum()
    total_trades_150 = buy_signals_150.sum()
    total_trades_200 = buy_signals_200.sum()
    trades_per_year = total_trades / total_years
    trades_per_year_150 = total_trades_150 / total_years
    trades_per_year_200 = total_trades_200 / total_years

    # 计算获胜次数
    winning_trades = (rai_data.loc[buy_signals, 'Future_Return_1Y'] > 0.10).sum()
    winning_trades_150 = (rai_data.loc[buy_signals_150, 'Future_Return_1Y'] > 0.10).sum()
    winning_trades_200 = (rai_data.loc[buy_signals_200, 'Future_Return_1Y'] > 0.10).sum()

    # 计算胜率
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    win_rate_150 = winning_trades_150 / total_trades_150 if total_trades_150 > 0 else 0
    win_rate_200 = winning_trades_200 / total_trades_200 if total_trades_200 > 0 else 0

    # 创建统计表格
    stats_table = html.Table([
        html.Thead(html.Tr([html.Th('指标'), html.Th('RAI < -100%'), html.Th('RAI < -150%'), html.Th('RAI < -200%')])),
        html.Tbody([
            html.Tr([html.Td('持有期'), html.Td('1年'), html.Td('1年'), html.Td('1年')]),
            html.Tr([html.Td('目标收益'), html.Td('>10%'), html.Td('>10%'), html.Td('>10%')]),
            html.Tr([html.Td('总交易次数'), html.Td(total_trades), html.Td(total_trades_150), html.Td(total_trades_200)]),
            html.Tr([html.Td('平均每年交易次数'), html.Td(f'{trades_per_year:.1f}'), html.Td(f'{trades_per_year_150:.1f}'), html.Td(f'{trades_per_year_200:.1f}')]),
            html.Tr([html.Td('获胜次数'), html.Td(winning_trades), html.Td(winning_trades_150), html.Td(winning_trades_200)]),
            html.Tr([html.Td('胜率'), html.Td(f'{win_rate:.1%}'), html.Td(f'{win_rate_150:.1%}'), html.Td(f'{win_rate_200:.1%}')]),
            html.Tr([html.Td('最大回撤'), html.Td(f"{calculate_max_drawdown(rai_data.loc[buy_signals, 'Future_Return_1Y']):.1%}"), 
                                      html.Td(f"{calculate_max_drawdown(rai_data.loc[buy_signals_150, 'Future_Return_1Y']):.1%}"),
                                      html.Td(f"{calculate_max_drawdown(rai_data.loc[buy_signals_200, 'Future_Return_1Y']):.1%}")])
        ])
    ], style={'border-collapse': 'collapse', 'width': '100%'})

    # 创建 RAI < -200% 期间的收益表格
    rai_below_200 = rai_data['World'] < -200
    events = []
    start_date = None
    for date, value in rai_below_200.items():
        if value and start_date is None:
            start_date = date
        elif not value and start_date is not None:
            end_date = date
            events.append((start_date, end_date))
            start_date = None

    event_returns = []
    for start, end in events:
        if end in rai_data.index:
            start_price = rai_data.loc[start, price_col]
            end_price = rai_data.loc[end, price_col]
            return_pct = (end_price - start_price) / start_price
            event_returns.append([start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), f'{return_pct:.1%}'])

    event_table = html.Table([
        html.Thead(html.Tr([html.Th('开始日期'), html.Th('结束日期'), html.Th('持有收益')])),
        html.Tbody([html.Tr([html.Td(start), html.Td(end), html.Td(return_pct)]) for start, end, return_pct in event_returns])
    ], style={'border-collapse': 'collapse', 'width': '100%'})

    return fig, stats_table, event_table

# 运行应用
if __name__ == "__main__":
    print("Dash 应用正在运行，请访问：http://127.0.0.1:8050/")
    app.run_server(debug=True)
