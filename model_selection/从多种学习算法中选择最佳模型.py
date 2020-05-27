import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# 设置随机数种子
np.random.seed(0)

# 加载数据
iris = datasets.load_iris()
features, target = iris.data, iris.target

# 创建流水线
pipe = Pipeline([("classifier", RandomForestClassifier(random_state=1, n_jobs=1))])

# 创建候选学习算法及超参数的字典
search_space = [{"classifier": [LogisticRegression(max_iter=3000, n_jobs=1, solver="liblinear")], "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier(random_state=1, n_jobs=1)],
                 "classifier__n_estimators": [10, 100, 1000], "classifier__max_features": [1, 2, 3]}]

grid_search = GridSearchCV(pipe, search_space, cv=5, scoring='neg_mean_squared_error', n_jobs=1).fit(features, target)

# 查看选择模型
print(grid_search.best_estimator_.get_params(deep=True)["classifier"])

# 预测目标向量
print(grid_search.predict(features))
