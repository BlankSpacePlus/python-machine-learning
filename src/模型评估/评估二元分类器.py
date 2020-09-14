from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 生成特征矩阵和目标向量
features, target = make_classification(n_samples=10000, n_features=3, n_informative=3, n_redundant=0, n_classes=2,
                                       random_state=1)

# 创建逻辑回归对象
logistic = LogisticRegression()

# 使用准确率对模型进行交叉验证
print(cross_val_score(logistic, features, target, scoring="accuracy"))

# 使用精确度对模型进行交叉验证
print(cross_val_score(logistic, features, target, scoring="precision"))

# 使用召回率
print(cross_val_score(logistic, features, target, scoring="recall"))

# 使用f1分数进行交叉验证
print(cross_val_score(logistic, features, target, scoring="f1"))

# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1,
                                                                            random_state=1)

# 对测试集进行预测
target_hat = logistic.fit(features_train, target_train).predict(features_test)

# 计算准确率
print(accuracy_score(target_test, target_hat))
