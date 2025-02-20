import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_subplot(df, start_year, ax1, ax2, title_prefix=""):
    """在指定的子图上创建分析图表"""
    # 筛选数据
    df_filtered = df[df.index >= f'{start_year}-01-01']
    
    # 计算周收益率，保留原始日期
    weekly_df = df_filtered.resample('W').last()
    weekly_df.index = df_filtered.resample('W').last().index  # 保留原始日期
    returns = weekly_df.pct_change().dropna()
    
    # 计算相关系数
    correlation = returns['STOXX50'].corr(returns['SP500'])
    
    # 绘制价格序列
    normalized_df = df_filtered / df_filtered.iloc[0] * 100
    
    # 计算过去12个月的数据
    one_year_ago = df_filtered.index[-1] - pd.DateOffset(months=12)
    recent_df = normalized_df[normalized_df.index >= one_year_ago]
    
    # 找出最近12个月内 STOXX 跑赢 SPX 的区间
    stoxx_outperform = recent_df['STOXX50'] > recent_df['SP500']
    changes = stoxx_outperform.astype(int).diff()
    start_dates = recent_df.index[changes == 1]
    end_dates = recent_df.index[changes == -1]
    
    # 处理边界情况
    if stoxx_outperform.iloc[0]:
        start_dates = pd.Index([recent_df.index[0]]).append(start_dates)
    if stoxx_outperform.iloc[-1]:
        end_dates = end_dates.append(pd.Index([recent_df.index[-1]]))
    
    # 绘制价格对比图
    ax1.plot(df_filtered.index, normalized_df['STOXX50'], label='STOXX 50', color='blue')
    ax1.plot(df_filtered.index, normalized_df['SP500'], label='S&P 500', color='red')
    
    # 添加阴影区域
    for start, end in zip(start_dates, end_dates):
        ax1.axvspan(start, end, color='peachpuff', alpha=0.3, 
                   label='STOXX跑赢SPX' if start == start_dates[0] else "")
    
    ax1.set_title(f'{title_prefix}STOXX 50 与 S&P 500 价格对比\n(起点设为100)')
    ax1.set_ylabel('指数')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 绘制相关性散点图
    ax2.scatter(returns['SP500'], returns['STOXX50'], alpha=0.5)
    
    # 标出最新点的位置
    latest_sp500_return = returns['SP500'].iloc[-1]
    latest_stoxx_return = returns['STOXX50'].iloc[-1]
    latest_date = returns.index[-1].strftime('%Y-%m-%d')
    
    ax2.scatter(latest_sp500_return, latest_stoxx_return, 
               color='red', s=100, label='当前位置')
    ax2.annotate(f'最新周位置\n{latest_date}\nSP500: {latest_sp500_return:.2%}\nSTOXX50: {latest_stoxx_return:.2%}',
                 xy=(latest_sp500_return, latest_stoxx_return),
                 xytext=(20, 20),
                 textcoords='offset points',
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('S&P 500 周收益率')
    ax2.set_ylabel('STOXX 50 周收益率')
    ax2.set_title(f'{title_prefix}周收益率散点图 (相关系数: {correlation:.4f})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 在原点添加参考线
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 设置散点图的x轴和y轴显示范围相等
    max_range = max(abs(ax2.get_xlim()[0]), abs(ax2.get_xlim()[1]), 
                   abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax2.set_xlim(-max_range, max_range)
    ax2.set_ylim(-max_range, max_range)
    
    return df_filtered, correlation

def analyze_correlations():
    try:
        # 读取CSV文件
        df = pd.read_csv('STOXX_SPX.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.columns = ['Date', 'STOXX50', 'SP500']
        df.set_index('Date', inplace=True)
        
        # 创建2x2的图表布局
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 2014年分析（左侧）
        ax1_2014 = fig.add_subplot(gs[0, 0])
        ax2_2014 = fig.add_subplot(gs[1, 0])
        df_2014, corr_2014 = create_subplot(df, 2014, ax1_2014, ax2_2014, "2014年至今 - ")
        
        # 2018年分析（右侧）
        ax1_2018 = fig.add_subplot(gs[0, 1])
        ax2_2018 = fig.add_subplot(gs[1, 1])
        df_2018, corr_2018 = create_subplot(df, 2018, ax1_2018, ax2_2018, "2018年至今 - ")
        
        plt.show()
        
        # 打印统计信息
        for year, df_filtered, corr in [(2014, df_2014, corr_2014), (2018, df_2018, corr_2018)]:
            print(f"\n{year}年至今分析结果：")
            print("-" * 50)
            print(f"时间范围: {df_filtered.index.min().strftime('%Y-%m-%d')} 到 {df_filtered.index.max().strftime('%Y-%m-%d')}")
            print(f"STOXX 50 和 S&P 500 的周收益率相关系数: {corr:.4f}")
            print(f"\n数据统计信息：")
            print("STOXX 50:")
            print(f"起始价格: {df_filtered['STOXX50'].iloc[0]:.2f}")
            print(f"结束价格: {df_filtered['STOXX50'].iloc[-1]:.2f}")
            print(f"总收益率: {(df_filtered['STOXX50'].iloc[-1]/df_filtered['STOXX50'].iloc[0] - 1):.2%}")
            print("\nS&P 500:")
            print(f"起始价格: {df_filtered['SP500'].iloc[0]:.2f}")
            print(f"结束价格: {df_filtered['SP500'].iloc[-1]:.2f}")
            print(f"总收益率: {(df_filtered['SP500'].iloc[-1]/df_filtered['SP500'].iloc[0] - 1):.2%}")
            print("-" * 50)
        
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 设置matplotlib支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    analyze_correlations()
