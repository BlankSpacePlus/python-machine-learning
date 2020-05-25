from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# 加载数据
iris = datasets.load_iris()
features = iris.data

# 标准化特征
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 创建MeanShift对象
cluster = AgglomerativeClustering(n_clusters=3)

# 训练模型
model = cluster.fit(features_std)

# 显示聚类情况
print(model.labels_)
