import dash
from dash import dcc, html
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from data_fetching import read_csv_file
from dash.dependencies import Input, Output

# 初始化 Dash 应用
app = dash.Dash(__name__)
app.title = 'RAI 可视化'

# 读取数据
rai_data, price_col = read_csv_file('RAI.csv')

# 数据预处理
rai_data['Date'] = rai_data['Date'].astype(str)
rai_data = rai_data[~rai_data['Date'].str.contains('Source|Produced', na=False)]
rai_data['World'] = pd.to_numeric(rai_data['World'], errors='coerce')
rai_data['World'] = rai_data['World'] * 100
rai_data['Date'] = pd.to_datetime(rai_data['Date'], format='%Y-%m-%d')
rai_data.set_index('Date', inplace=True)

# 计算标普500未来3个月下跌超过10%的区间
crash_periods = []
if price_col in rai_data.columns:
    future_window = 63
    future_returns = []
    for i in range(len(rai_data)):
        if i + future_window < len(rai_data):
            future_return = (rai_data[price_col].iloc[i+future_window] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i]
            future_returns.append(future_return)
        else:
            future_returns.append(np.nan)
    rai_data['Future_Return'] = future_returns
    crash_start = rai_data['Future_Return'] < -0.10
    for i in range(len(rai_data)):
        if crash_start.iloc[i]:
            start_date = rai_data.index[i]
            for j in range(i+1, min(i+future_window+1, len(rai_data))):
                if (rai_data[price_col].iloc[j] - rai_data[price_col].iloc[i]) / rai_data[price_col].iloc[i] < -0.10:
                    end_date = rai_data.index[j]
                    crash_periods.append((start_date, end_date))
                    break

# 创建布局
app.layout = html.Div([
    html.H1('Risk Appetite Indicator (RAI) 可视化'),
    dcc.Graph(id='rai-graph'),
    dcc.Graph(id='sp500-graph'),
    html.Div(id='latest-values')
])

# 回调函数
@app.callback(
    [Output('rai-graph', 'figure'),
     Output('sp500-graph', 'figure'),
     Output('latest-values', 'children')],
    [Input('rai-graph', 'relayoutData')]
)
def update_graphs(relayoutData):
    # RAI 图表
    rai_trace = go.Scatter(
        x=rai_data.index,
        y=rai_data['World'],
        mode='lines',
        name='RAI',
        line=dict(color='blue')
    )

    shapes = []
    for start, end in crash_periods:
        shapes.append(dict(
            type='rect',
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            yref='paper',
            fillcolor='peachpuff',
            opacity=0.3,
            layer='below',
            line_width=0
        ))

    rai_layout = go.Layout(
        title='Risk Appetite Indicator (RAI) 时间序列图',
        yaxis=dict(title='RAI 值 (%)', tickformat='.0%'),
        xaxis=dict(title='时间'),
        shapes=shapes,
        hovermode='x unified'
    )

    rai_figure = go.Figure(data=[rai_trace], layout=rai_layout)

    # 标普500图表
    if price_col in rai_data.columns:
        sp500_trace = go.Scatter(
            x=rai_data.index,
            y=rai_data[price_col],
            mode='lines',
            name='标普500价格',
            line=dict(color='green')
        )

        sp500_layout = go.Layout(
            title='标普500价格走势',
            yaxis=dict(title='价格'),
            xaxis=dict(title='时间'),
            shapes=shapes,
            hovermode='x unified'
        )

        sp500_figure = go.Figure(data=[sp500_trace], layout=sp500_layout)
    else:
        sp500_figure = go.Figure()

    # 最新值显示
    latest_date = rai_data.index[-1]
    latest_rai = rai_data['World'].iloc[-1]
    latest_sp500 = rai_data[price_col].iloc[-1] if price_col in rai_data.columns else None

    latest_values = html.Div([
        html.H3('最新值：'),
        html.P(f'日期: {latest_date.strftime("%Y-%m-%d")}'),
        html.P(f'RAI 值: {latest_rai:.1f}%'),
        html.P(f'标普500价格: {latest_sp500:.2f}' if latest_sp500 else '无标普500数据')
    ])

    return rai_figure, sp500_figure, latest_values

# 运行应用
if __name__ == '__main__':
    print("\nDash 应用正在启动...")
    print("请在浏览器中访问以下网址：")
    print("http://127.0.0.1:8050/")
    print("按 Ctrl+C 停止服务器\n")
    app.run_server(debug=True) 