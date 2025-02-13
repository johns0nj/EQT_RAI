import pandas as pd
import yfinance as yf  # 需要安装 yfinance 库

def read_csv_file(file_path):
    """
    读取CSV文件并返回DataFrame对象
    """
    # 读取 CSV 文件，跳过标题行
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['Date', 'World'])

    # 打印原始数据前几行
    print("\n原始数据预览：")
    print(df.head())

    # 定义一个函数，将括号表示的负数转换为标准负数
    def convert_negative(value):
        try:
            if isinstance(value, str):
                # 去掉逗号和空格
                value = value.replace(',', '').strip()
                # 处理括号表示的负数
                if value.startswith('(') and value.endswith(')'):
                    return -float(value[1:-1])
                # 处理负号表示的负数
                elif value.startswith('-'):
                    return float(value)
                # 处理其他可能的负数表示方式
                elif value.endswith('-'):
                    return -float(value[:-1])
            # 如果不是字符串，直接尝试转换为浮点数
            return float(value)
        except (ValueError, TypeError) as e:
            print(f"无法转换值: {value}, 错误: {e}")
            return value

    # 应用转换函数到 'World' 列
    df['World'] = df['World'].apply(convert_negative)

    # 确保 World 列是数值类型
    df['World'] = pd.to_numeric(df['World'], errors='coerce')

    # 打印转换后的数据前几行
    print("\n转换后数据预览：")
    print(df.head())

    # 打印负值数量
    print(f"\n负值数量：{(df['World'] < 0).sum()}")
    if (df['World'] < 0).sum() > 0:
        print("负值示例：")
        print(df[df['World'] < 0].head())
    else:
        print("未发现负值，请检查原始数据格式")

    # 获取标普500数据
    # 首先尝试将日期转换为datetime格式
    try:
        # 先尝试使用指定格式，包含时间部分
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            # 如果失败，尝试只解析日期部分
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        except ValueError:
            try:
                # 如果仍然失败，尝试自动推断格式
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            except Exception as e:
                print(f"日期解析失败: {e}")
                # 如果仍然失败，尝试其他可能的格式
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # 检查是否有无效日期
    if df['Date'].isnull().any():
        print("警告：发现无效日期，请检查原始数据")
        print(df[df['Date'].isnull()])
    
    # 获取日期范围
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    # 从 Yahoo Finance 获取标普500数据
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    
    # 处理多级列索引
    if isinstance(sp500.columns, pd.MultiIndex):
        # 将多级列索引转换为单级，只保留第一级
        sp500.columns = sp500.columns.get_level_values(0)
    
    # 使用 'Close' 列代替 'Adj Close'
    price_col = 'Close'  # 直接使用 'Close'
    
    # 只保留收盘价
    sp500 = sp500[[price_col]]
    
    # 重置索引以便合并
    sp500.reset_index(inplace=True)
    sp500.rename(columns={'Date': 'Date', price_col: price_col}, inplace=True)
    
    # 确保日期格式一致
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 合并数据
    df = pd.merge(df, sp500[['Date', price_col]], on='Date', how='left')

    # 将标普500每日价格数据保存到SP500.csv
    sp500[['Date', price_col]].to_csv('SP500.csv', index=False, header=['DATE', 'PRICE'])

    return df, price_col  # 返回 price_col

def main():
    file_path = 'RAI.csv'
    df, price_col = read_csv_file(file_path)  # 接收 price_col
    
    # 打印数据信息
    print("\n数据预览：")
    print(df.head(20))

    print("\n数值范围：")
    print(f"最小值：{df['World'].min()}")
    print(f"最大值：{df['World'].max()}")
    
    # 打印标普500数据信息
    print("\n标普500数据统计：")
    print(df[price_col].describe())  # 现在可以访问 price_col
    
    return df

if __name__ == "__main__":
    main()