from keras import models
from keras import layers

# 启动神经网络
network = models.Sequential()

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation='relu', input_shape=(10, )))
network.add(layers.Dense(units=16, activation='relu'))

# 使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1, activation='sigmoid'))

# 编译神经网络
network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
