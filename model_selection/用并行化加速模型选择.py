import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 创建逻辑回归对象
logistic = LogisticRegression(max_iter=10000)

# 创建正则化惩罚的超参数候选值区间
penalty = ["l1", "l2"]

# 创建参数C的候选区间
C = np.logspace(0, 4, 100)

# 创建超参数选项
hyper_parameters = dict(C=C)

# 创建网格搜索对象 n_jobs=-1
grid_search = GridSearchCV(logistic, hyper_parameters, cv=5, verbose=1, n_jobs=1)

# 执行网格搜索
beat_model = grid_search.fit(features, target)
