# 导入必要的库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载
def load_data(file_path):
    """
    加载数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查文件路径")
    data = pd.read_csv(file_path)
    return data

# 2. 数据预处理
def preprocess_data(data):
    """
    数据清洗和预处理
    """
    # 示例：填充缺失值
    data.fillna(method='ffill', inplace=True)
    return data

# 3. 策略实现
def implement_strategy(data):
    """
    实现具体策略
    """
    # 示例：简单移动平均策略
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # 生成交易信号
    data['Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1)
    return data

# 4. 结果可视化
def visualize_results(data):
    """
    可视化策略结果
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['SMA_50'], label='50-Day SMA')
    plt.plot(data['SMA_200'], label='200-Day SMA')
    plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], marker='^', color='g', label='Buy Signal')
    plt.scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], marker='v', color='r', label='Sell Signal')
    plt.legend()
    plt.title('Trading Strategy Results')
    plt.show()

# 主函数
def main():
    # 数据路径
    file_path = 'data.csv'
    
    try:
        # 加载数据
        data = load_data(file_path)
        
        # 数据预处理
        data = preprocess_data(data)
        
        # 实现策略
        data = implement_strategy(data)
        
        # 可视化结果
        visualize_results(data)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
