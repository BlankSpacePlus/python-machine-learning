from sklearn.datasets import make_blobs

# 生成特征矩阵、目标向量以及模型的系数
features, target = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)

# 查看特征矩阵和目标向量
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])
