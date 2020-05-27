import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 加载数据
iris = datasets.load_iris()
features, target = iris.data, iris.target

# 创建目标分类的名称列表
class_names = iris.target_names

# 训练训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)

# 创建逻辑回归分类模型
classifier = LogisticRegression(max_iter=3000)

# 训练模型并做出预测
target_predicted = classifier.fit(features_train, target_train).predict(features_test)

# 创建混淆矩阵
matrix = confusion_matrix(target_test, target_predicted)

# 创建DataFrame
data_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)

# 保存一下热力图
with PdfPages("分类器性能可视化热力图.pdf") as pdf:
    fig = plt.figure()
    sns.heatmap(data_frame, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("True Class"), plt.ylabel("Predict Class")
    plt.tight_layout()
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)
