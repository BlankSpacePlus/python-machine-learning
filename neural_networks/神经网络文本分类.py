import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models, layers

# 设定随机种子
np.random.seed(0)

# 设定想要的特征数量
number_of_features = 1000

# 从影评中加载数据和目标向量
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

# 采用自动填充值或者截断的方式，使每个样本都有400个特征
features_train = sequence.pad_sequences(data_train, maxlen=400)
features_test = sequence.pad_sequences(data_test, maxlen=400)

# 创建神经网络对象
network = models.Sequential()

# 添加嵌入层
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))

# 添加一个有128个神经元长短记忆网络层
network.add(layers.LSTM(units=128))

# 添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

# 训练神经网络
history = network.fit(features_train, target_train, epochs=3, verbose=1, batch_size=100,
                      validation_data=(features_test, target_test))

# 查看第一个训练样本
print(data_train[0])

# 查看第一个测试样本
print(data_test[0])
