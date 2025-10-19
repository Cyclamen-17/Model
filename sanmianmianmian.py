import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据加载
folder_path = "C:/Users/36288/Desktop/stock/stock data"  # 替换为实际文件夹路径
all_data = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df_single = pd.read_csv(file_path)
        all_data = pd.concat([all_data, df_single], ignore_index=True)

# 检查数据是否加载成功
if len(all_data) == 0:
    print("未找到CSV文件或文件为空，请检查文件夹路径和文件内容！")
else:
    print(all_data.head())
    print(f"合并后的数据总行数：{len(all_data)}")

    # 确保数据中有ts_code和close列（根据实际列名修改）
    if "ts_code" not in all_data.columns or "close" not in all_data.columns:
        print("数据中缺少ts_code或close列，请检查CSV文件的列名！")
    else:
        # 2. 板块联动分析
        def calculate_correlation(df):
            pivot_close = df.pivot(columns="ts_code", values="close")
            corr_matrix = pivot_close.corr()
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    stock1 = corr_matrix.columns[i]
                    stock2 = corr_matrix.columns[j]
                    corr = corr_matrix.iloc[i, j]
                    corr_pairs.append((stock1, stock2, corr))
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            return corr_pairs, corr_matrix

        corr_pairs, corr_matrix = calculate_correlation(all_data)
        print("前5对高相关股票：")
        for pair in corr_pairs[:5]:
            print(f"股票{pair[0]}与股票{pair[1]}，相关系数：{pair[2]:.4f}")

        # 可视化相关系数矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, annot=False)
        plt.title("股票收盘价相关系数矩阵")
        plt.show()

        # 3. 板块分类（K-means聚类）
        def stock_clustering(df, n_clusters=3):
            pivot_close = df.pivot(columns="ts_code", values="close")
            daily_return = pivot_close.pct_change().dropna()
            features = daily_return.T
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
            stock_cluster = {stock: label for stock, label in zip(features.index, kmeans.labels_)}
            return stock_cluster

        stock_cluster = stock_clustering(all_data)
        print("股票板块分类结果：")
        for stock, cluster in stock_cluster.items():
            print(f"股票{stock} 归属板块{cluster}")

        # 4. 板块周期性分析（以板块0为例）
        def analyze_cycle(df, stock_cluster, cluster_id=0, period=20):
            cluster_stocks = [stock for stock, c in stock_cluster.items() if c == cluster_id]
            pivot_close = df.pivot(columns="ts_code", values="close")[cluster_stocks]
            cluster_avg_close = pivot_close.mean(axis=1)
            decomposition = seasonal_decompose(cluster_avg_close, model="additive", period=period)
            plt.figure(figsize=(12, 8))
            plt.subplot(411)
            plt.plot(cluster_avg_close, label="原始序列")
            plt.legend()
            plt.subplot(412)
            plt.plot(decomposition.trend, label="趋势项")
            plt.legend()
            plt.subplot(413)
            plt.plot(decomposition.seasonal, label="周期项")
            plt.legend()
            plt.subplot(414)
            plt.plot(decomposition.resid, label="残差项")
            plt.legend()
            plt.tight_layout()
            plt.title(f"板块{cluster_id}股价周期分解")
            plt.show()

        analyze_cycle(all_data, stock_cluster)