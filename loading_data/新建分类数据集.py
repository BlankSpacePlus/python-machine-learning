from sklearn.datasets import make_classification

# 生成特征矩阵、目标向量以及模型的系数
features, target = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0, n_classes=2,
                                       weights=[.25, .75], random_state=1)

# 查看特征矩阵和目标向量
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
