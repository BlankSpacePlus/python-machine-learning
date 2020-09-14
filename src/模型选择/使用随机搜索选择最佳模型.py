from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建逻辑回归对象
logistic = linear_model.LogisticRegression(max_iter=10000)

# 创建正则化惩罚的候选超参数区间
penalty = ['l1', 'l2']

# 创建正则化惩罚的候选超参数区间
C = uniform(loc=0, scale=4)

# 创建超参数字典
hyper_parameters = dict(C=C)

# 创建随机搜索对象 n_jobs=-1
randomized_search = RandomizedSearchCV(logistic, hyper_parameters, random_state=1, n_iter=100, cv=5)

# 训练随机搜索
best_model = randomized_search.fit(features, target)

# 定义区间 (0,4) 上的均匀分布，并从中抽取10个样本值
print(uniform(loc=0, scale=4).rvs(10))

# 查看最佳超参数
print('Best Penalty: ', best_model.best_estimator_.get_params()['penalty'])
print('Best C: ', best_model.best_estimator_.get_params()['C'])

# 预测目标向量
print(best_model.predict(features))
