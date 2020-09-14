from sklearn.feature_selection import VarianceThreshold

'''
用如下信息创建特征矩阵
特征0：80% 为分类 0
特征1：80% 为分类 1
特征2：60% 为分类 0，40% 为分类 1
'''
features = [[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]

# 创建VarianceThreshold对象并运行
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
print(thresholder.fit_transform(features))
