from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成特征矩阵和目标向量
features, target = make_classification(n_samples=10000, n_features=3, n_informative=3, n_redundant=0, n_classes=3,
                                       random_state=1)

# 创建逻辑回归对象
logistic = LogisticRegression()

# 使用准确率对模型进行交叉验证
print(cross_val_score(logistic, features, target, scoring="accuracy"))

# 计算准确率
print(cross_val_score(logistic, features, target, scoring='f1_macro'))
