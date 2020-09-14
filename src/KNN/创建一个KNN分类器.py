from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 创建standardizer
standardizer = StandardScaler()

# 标准化特征
x_std = standardizer.fit_transform(x)

# 训练一个有5个邻居的KNN分类器
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(x_std, y)

# 创建两个观察值
new_observation = [[0.75, 0.75, 0.75, 0.75], [1, 1, 1, 1]]

# 预测这两个观察值的分类
print(knn.predict(new_observation))

# 查看每个观察值分别属于三个分类中的某一个的概率
print(knn.predict_proba(new_observation))
