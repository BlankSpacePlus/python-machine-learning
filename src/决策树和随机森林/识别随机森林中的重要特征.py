import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.datasets import load_iris
from matplotlib.backends.backend_pdf import PdfPages

# 加载数据
iris = load_iris()
features = iris.data
target = iris.target

# 创建随机森林分类器对象
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# 训练模型
model = random_forest.fit(features, target)

# 计算特征值的重要性
importance = model.feature_importances_

# 查看特征值的重要性
print(importance)

# 将特征的重要性按降序排列
indices = np.argsort(importance)[::-1]

# 按照特征的重要性对特征名称重新排序
names = [iris.feature_names[i] for i in indices]

with PdfPages("random_forest_importance.pdf") as pdf:
    # 创建图
    fig = plt.figure()
    # 创建图标题
    plt.title("Feature Importance")
    # 添加数据条
    plt.bar(range(features.shape[1]), importance[indices])
    # 将特征名称添加为x轴标签
    plt.xticks(range(features.shape[1]), names, rotation=90)
    # 显示图
    plt.show()
    # 保存当前config
    pdf.savefig(fig)
    # 关闭当前config
    plt.close(fig)
