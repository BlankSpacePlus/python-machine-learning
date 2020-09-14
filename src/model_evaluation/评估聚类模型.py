import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

features, _ = make_blobs(n_samples=1000, n_features=10, centers=2, cluster_std=0.5, shuffle=True, random_state=1)

# 使用KMean方法对数据聚类，以预测其分类
model = KMeans(n_clusters=2, random_state=1).fit(features)

# 获取预测的分类
target_predicted = model.labels_

# 评估模型
print(silhouette_score(features, target_predicted))
