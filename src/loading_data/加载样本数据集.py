from sklearn import datasets

'''
加载已有样本数据集，这里是手写数字数据集load_digits
'''

# 加载手写数字数据集
digits = datasets.load_digits()

# 创建特征矩阵
feature = digits.data

# 创建目标向量
target = digits.target

# 查看第一个样本数据
print(feature[0])
