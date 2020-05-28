import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models, layers

# 设定随机种子
np.random.seed(0)

# 设定想要的特征数量
number_of_features = 5000

# 从影评中加载数据和目标向量
(data_train, target_vector_train), (data_test, target_vector_test) = reuters.load_data(num_words=number_of_features)

# 将影评数据转化为one-hot编码过的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# 将one-hot编码的特征数据转换成特征矩阵
target_train = to_categorical(target_vector_train)
target_test = to_categorical(target_vector_test)

# 创建神经网络对象
network = models.Sequential()

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=100, activation="relu", input_shape=(number_of_features, )))

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=100, activation="relu"))

# 添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=46, activation="softmax"))

# 编译神经网络
network.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 训练神经网络
history = network.fit(features_train, target_train, epochs=3, verbose=1, batch_size=100,
                      validation_data=(features_test, target_test))

# 查看目标矩阵
print(target_train)
