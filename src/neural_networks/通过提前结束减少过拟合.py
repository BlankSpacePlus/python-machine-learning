import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 设定随机种子
np.random.seed(0)

# 设定想要的特征数量
number_of_features = 1000

# 从影评中加载数据和目标向量
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

# 将影评数据转化为one-hot编码过的特征矩阵
tokenizer = Tokenizer(num_words=number_of_features)
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

# 创建神经网络对象
network = models.Sequential()

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features, )))

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu"))

# 添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1, activation="sigmoid"))

# 编译神经网络
network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 设置一个回调函数来提前结束训练，并保存训练结束时的最佳模型
callbacks = [EarlyStopping(monitor="val_loss", patience=2),ModelCheckpoint(filepath="callback_models_1.hdf5")]

# 训练神经网络
history = network.fit(features_train, target_train, epochs=20, callbacks=callbacks, verbose=1, batch_size=100,
                      validation_data=(features_test, target_test))
