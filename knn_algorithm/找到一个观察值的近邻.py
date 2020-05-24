from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()
features = iris.data

# 创建standardizer
standardizer = StandardScaler()

# 特征标准化
features_standardized = standardizer.fit_transform(features)

# 两个最近的观察值
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

# 创建一个观察值
new_observation = [1, 1, 1, 1]

# 获取离观察值最近的两个观察值的索引，以及到这两个点之间的距离
distances, indices = nearest_neighbors.kneighbors([new_observation])

# 查看最近的两个观察值
print(features_standardized[indices])

# 找到按照欧氏距离来计算最近的两个邻居
nearest_neighbors_euclidean = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(features_standardized)

# 每个观察值和它最近的3个邻居的列表（包括它自己）
nearest_neighbors_euclidean_with_self = nearest_neighbors_euclidean.kneighbors_graph(features_standardized).toarray()

# 从最临近邻居的列表里移除自己
for i, x in enumerate(nearest_neighbors_euclidean_with_self):
    x[i] = 0

# 查看离第一个观察值最近的两个邻居
print(nearest_neighbors_euclidean_with_self[0])
