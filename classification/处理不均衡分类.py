import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

'''
我们可能需要处理一个分类极度不均衡的目标向量。
首先可以尝试收集更多的数据。
如果做不到，就改变评估模型的衡量标准。
如果这也不起作用，可以考虑使用嵌入分类权重参数（如果有的话）的模型、上采样或下采样。
'''
# 加载鸢尾花数据
iris = load_iris()

# 创建特征矩阵
features = iris.data

# 创建目标向量
target = iris.target

# 移除前40个观察值
features = features[40:, :]
target = target[40:]

# 创建二元目标向量来标识观察值是否为类别0
target = np.where((target == 0), 0, 1)

# 查看不均衡的目标向量
print(target)

# 创建权重
weights = {0: .9, 1: 0.1}

# 创建带权重的随机森林分类器
RandomForestClassifier(class_weight=weights)

# 还可以传参balanced，它会自动创建与分类的频数成反比的权重(下面的支持均衡分类权重)
RandomForestClassifier(class_weight="balanced")

# 给每个分类的观察值打标签
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# 确认每个分类的观察值数量
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# 对于每个分类为0的观察值，从分类为1的数据中进行无放回的随机采样
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# 将分类为0的特征矩阵和下采样的分类为1的目标向量连接起来
print(np.hstack((target[i_class0], target[i_class1_downsampled])))

# 将分类为0的特征矩阵和下采样的分类为1的特征矩阵连接起来
print(np.vstack((features[i_class0, :], features[i_class1_downsampled, :]))[0:5])

# 对于每个分类为1的观察值，从分类为0的数据中进行又放回的随机抽样
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# 将上采样得到的分类为0的目标向量和分类为1的目标向量连接起来
print(np.concatenate((target[i_class0_upsampled], target[i_class1])))

# 将上采样得到的分类为0的目标向量和分类为1的特征矩阵连接起来
print(np.vstack((features[i_class0_upsampled, :], features[i_class1, :]))[0:5])
