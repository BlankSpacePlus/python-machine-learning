from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建一对多的逻辑回归对象
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")

# 训练模型
model = logistic_regression.fit(features_standardized, target)

new_observation = [[.5, .5, .5, .5]]
print(model.predict(new_observation))
print(model.predict_proba(new_observation))
