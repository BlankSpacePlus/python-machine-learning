import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置随机数种子
np.random.seed(0)

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 创建一个包含StandardScaler和PCA的预处理对象
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# 创建一个流水线
pipe = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(solver="liblinear", max_iter=10000))])

# 创建候选值的取值空间
search_space = [{"preprocess__pca__n_components": [1, 2, 3], "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# 创建网格搜索对象 n_jobs=-1
clf = GridSearchCV(pipe, search_space, cv=5)

# 创建网格搜索对象
best_model = clf.fit(features, target)

# 查看最佳模型的主成分数量
print(best_model.best_estimator_.get_params()['preprocess__pca__n_components'])
