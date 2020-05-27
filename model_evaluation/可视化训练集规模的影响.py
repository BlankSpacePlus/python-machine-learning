import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

# 架子啊数据
digits = load_digits()
features, target = digits.data, digits.target

# 使用交叉验证为不同规模的训练集计算训练和测试得分 n_jobs=-1
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), features, target, cv=10,
                                                        scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))

# 计算训练集得分的平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# 计算测试集得分的平均值和标准差
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 运行比较慢，所以保存一下
with PdfPages("训练集规模的影响可视化图.pdf") as pdf:
    fig = plt.figure()
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)

