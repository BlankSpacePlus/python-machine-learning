import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model

'''
Keras不推荐使用pickle格式保存模型，而是要将模型保存为HDF5文件。
HDF5文件包含了我们所需要的一切，不仅包含加载数据模型做预测所需要的结构和训练后的参数，而且包含重新训练所需要的各种设置（即损失、优化器的设置和当前状态）
'''

# 设置随机种子
np.random.seed(0)

# 设定想要的特征数量
number_of_features = 1000

# 从影评中架子啊数据和目标向量
(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)

# 把影评数据转换为one-hot编码的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")

# 启动神经网络
network = models.Sequential()

# 添加使用ReLu激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features, )))

# 添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络(交叉熵、均方根传播、准确率作为性能指标)
network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 训练神经网络
history = network.fit(train_features, train_target, epochs=3, verbose=0, batch_size=100,
                      validation_data=(test_features, test_target))

# 保存神经网络
network.save("model.h5")

# 加载神经网路
network = load_model("model.h5")
