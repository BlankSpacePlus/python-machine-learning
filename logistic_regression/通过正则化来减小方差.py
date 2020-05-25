from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 标准化特征
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# 创建一个决策树分类器对象 （！！！！！使用 n_jobs=-1 会报错）
logistic_regression = LogisticRegressionCV(penalty='l2', Cs=10, random_state=0)

# 训练模型
model = logistic_regression.fit(features_standardized, target)
