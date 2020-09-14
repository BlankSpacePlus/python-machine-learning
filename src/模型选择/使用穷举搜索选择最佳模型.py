import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建逻辑回归对象
logistic = linear_model.LogisticRegression(solver="liblinear")

# 创建正则化惩罚的候选超参数区间
penalty = ['l1', 'l2']

# 创建正则化候选超参数区间
C = np.logspace(0, 4, 10)

# 创建候选超参数的字典 , penalty=penalty
hyper_parameters = dict(C=C)

# 创建网格搜索对象
grid_search = GridSearchCV(logistic, hyper_parameters, cv=5)

# 训练网格搜索
best_model = grid_search.fit(features, target)

print(np.logspace(0, 4, 10))

# 查看最佳超参数
# l1 ？？
print('Best Penalty: ', best_model.best_estimator_.get_params()['penalty'])
print('Best C: ', best_model.best_estimator_.get_params()['C'])

# 预测目标向量
print(best_model.predict(features))
