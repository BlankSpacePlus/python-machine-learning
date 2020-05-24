from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建standardizer
standardizer = StandardScaler()

# 标准化特征
features_standardized = standardizer.fit_transform(features)

# 创建一个基于半径的最近邻分类器
rnn = RadiusNeighborsClassifier(radius=.5, n_jobs=-1).fit(features_standardized, target)

# 创建两个观察值
new_observation = [[1, 1, 1, 1]]

# 预测这两个观察值的分类
print(rnn.predict(new_observation))
