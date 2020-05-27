import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 创建逻辑回归对象
logistic = LogisticRegression(max_iter=10000)

# 创建超参数C的20个候选值
C = np.logspace(0, 4, 20)

# 创建超参数字典
hyper_parameters = dict(C=C)

# 创建网格搜索
grid_search = GridSearchCV(logistic, hyper_parameters, cv=5)

# 执行嵌套交叉验证并输出平均得分
print(cross_val_score(grid_search, features, target).mean())

grid_search = GridSearchCV(logistic, hyper_parameters, cv=5, verbose=1)
best_model = grid_search.fit(features, target)

scores = cross_val_score(grid_search, features, target)
