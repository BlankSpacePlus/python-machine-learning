import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# 创建特征
feature = np.array([["Texas"], ["California"], ["Texas"], ["Delaware"], ["Texas"]])

# 创建one-hot编码器
one_hot = LabelBinarizer()

# 对特征进行one-hot编码
print(one_hot.fit_transform(feature))

# 查看特征的分类
print(one_hot.classes_)

# 对onr-hot编码进行逆转换
print(one_hot.inverse_transform(one_hot.transform(feature)))

# 创建虚拟变量
print(pd.get_dummies(feature[:, 0]))

# 创建有多个分类的特征
multiclass_feature = [("Texas", "Florida"), ("California", "Alabama"), ("Texas", "Florida"), ("Delware", "Florida"),
                      ("Texas", "Alabama")]

# 创建能处理多个分类的one-hot编码器
one_hot_multiclass = MultiLabelBinarizer()

# 对特征进行onr-hot编码
print(one_hot_multiclass.fit_transform(multiclass_feature))

# 查看分类
print(one_hot_multiclass.classes_)
