from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 加载仅有两个分类的数据
iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建一个逻辑回归的对象
logistic_regression = LogisticRegression(random_state=0)

# 训练模型
model = logistic_regression.fit(features_standardized, target)

# 创建一个新的观察值
new_observation = [[.5, .5, .5, .5]]

# 预测分类
print(model.predict(new_observation))

# 查看预测的概率
print(model.predict_proba(new_observation))
