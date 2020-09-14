import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

'''
识别样本中的一些极端观察值（异常值）
'''

# 创建模拟数据
features, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=1)

# 将第一个观察值的值替换为极端值
features[0, 0] = 10000
features[0, 1] = 10000

# 创建识别器
outlier_detector = EllipticEnvelope(contamination=.1)
'''
contamination的选择是一个玄学问题，可以认为是你自己估计的数据的清洁程度
如果你认为数据只有很少的异常值时，contamination可以设置的比较小
如果你认为可能存在好多个异常值时，contamination可以设置的比较大
'''

# 拟合识别器
outlier_detector.fit(features)

# 预测异常值
print(outlier_detector.predict(features))

'''
除了查看所有的特征值，我们还可以只查看某些特征，并使用四分位差（IQR）来识别这些特征的极端值
'''
feature = features[:, 0]


# 创建一个函数来返回异常值的下标
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where(x > upper_bound) | (x < lower_bound)


# 执行函数
print(indicies_of_outliers(feature))
