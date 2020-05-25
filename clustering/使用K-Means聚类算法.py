from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 加载数据
iris = datasets.load_iris()
features = iris.data

# 标准化特征
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# 创建K-Means对象 n_jobs=-1
cluster = KMeans(n_clusters=3, random_state=0)

# 训练模型
model = cluster.fit(features_std)

# 查看预测分类
print(model.labels_)

# 查看真实分类 ~ KMeans做的不错！
print(iris.target)

# 创建新的观察值
new_observation = [[0.8, 0.8, 0.8, 0.8]]

# 预测观察值的分类
print(model.predict(new_observation))

# 查看分类的中心点
print(model.cluster_centers_)
