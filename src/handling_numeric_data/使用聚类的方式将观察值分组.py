import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 创建模拟的特征矩阵
features, _ = make_blobs(n_samples=50, n_features=2, centers=3, random_state=1)

# 创建数据帧
data_frame = pd.DataFrame(features, columns=["feature1", "feature2"])

# 创建K均值聚类器
clusterer = KMeans(3, random_state=0)

# 将聚类器应用在特征上
clusterer.fit(features)

# 预测聚类的值
data_frame["group"] = clusterer.predict(features)

# 查看几个观察值
print(data_frame.head(5))
