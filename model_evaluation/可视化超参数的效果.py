import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

# 加载数据
digits = load_digits()
features, target = digits.data, digits.target

# 创建参数的变化范围
param_range = np.arange(1, 250, 2)

# 对区间内的参数值分别计算模型在训练集和测试集上的准确率 n_jobs=-1
train_scores, test_sores = validation_curve(RandomForestClassifier(), features, target, param_name="n_estimators",
                                            param_range=param_range, cv=3, scoring="accuracy")

# 计算模型在训练集上得分的平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# 计算模型在测试集上得分的平均值和标准差
test_mean = np.mean(test_sores, axis=1)
test_std = np.std(test_sores, axis=1)

# 运行比较慢，所以保存一下
with PdfPages("超参数值效果可视化图.pdf") as pdf:
    fig = plt.figure()
    plt.plot(param_range, train_mean, '--', color="#111111", label="Training score")
    plt.plot(param_range, test_mean, color="#111111", label="Cross-validation score")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    plt.title("Validation Curve with Random Forest")
    plt.xlabel("Number Of Trees"), plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)
