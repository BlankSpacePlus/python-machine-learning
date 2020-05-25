from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# 加载数据
iris = datasets.load_iris()
features = iris.data

# 标准化特征
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 创建K-Means对象
cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)

# 训练模型
model = cluster.fit(features_std)
