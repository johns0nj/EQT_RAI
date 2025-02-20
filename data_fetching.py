import pandas as pd

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

    # 读取标普500数据
    try:
        sp500 = pd.read_csv('SP500.csv')
        sp500['DATE'] = pd.to_datetime(sp500['DATE'])
        sp500.rename(columns={'DATE': 'Date', 'PRICE': 'Close'}, inplace=True)
    except Exception as e:
        print(f"读取标普500数据失败: {e}")
        sp500 = pd.DataFrame(columns=['Date', 'Close'])

    # 将日期转换为datetime格式
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    except ValueError:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
        except Exception as e:
            print(f"日期解析失败: {e}")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 检查是否有无效日期
    if df['Date'].isnull().any():
        print("警告：发现无效日期，请检查原始数据")
        print(df[df['Date'].isnull()])

    # 合并数据
    df = pd.merge(df, sp500[['Date', 'Close']], on='Date', how='left')

    return df, 'Close'  # 返回 price_col

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