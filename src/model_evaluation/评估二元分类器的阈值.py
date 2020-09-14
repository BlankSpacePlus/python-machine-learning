import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score

# 生成特征矩阵和目标向量
features, target = make_classification(n_samples=10000, n_features=10, n_informative=3, n_classes=2, random_state=3)

# 将样本划分为测试集和训练集
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1,
                                                                            random_state=1)

# 创建逻辑回归对象
logistic = LogisticRegression()

# 训练模型
logistic.fit(features_train, target_train)

# 预测获取的概率
target_probabilities = logistic.predict_proba(features_test)[:, 1]

# 计算真阳性和假阳性的概率
false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilities)

# 画出ROC曲线
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

# 获取预测的概率
print(logistic.predict_proba(features_test)[0:1])

# 使用classes_查看分类
print(logistic.classes_)

# 50%的概率阈值
print("Threshold: ", threshold[116])
print("True Positive Rate: ", true_positive_rate[116])
print("False Positive Rate: ", false_positive_rate[116])

# 80%的概率阈值
print("Threshold: ", threshold[45])
print("True Positive Rate: ", true_positive_rate[45])
print("False Positive Rate: ", false_positive_rate[45])

# 计算ORC曲线下方的面积
print(roc_auc_score(target_test, target_probabilities))
