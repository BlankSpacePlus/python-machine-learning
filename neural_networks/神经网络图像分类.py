import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k

# 设置彩色通道优先
k.set_image_data_format("channels_first")

# 设置随机种子
np.random.seed(0)

# 图像信息
channels = 1
height = 28
width = 28

# 从MNIST数据集中读取数据和目标
(data_train, target_train), (data_test, target_test) = mnist.load_data()

# 将训练集图像数据转换成特征
data_train = data_train.reshape(data_train.shape[0], channels, height, width)

# 将测试集图像数据转换成特征
data_test = data_test.reshape(data_test.shape[0], channels, height, width)

# 将像素的强度值收缩到0和1之间
features_train = data_train / 255
features_test = data_test / 255

# 对目标进行one-hot编码
target_train = np_utils.to_categorical(target_train)
target_test = np_utils.to_categorical(target_test)
number_of_classes = target_test.shape[1]

# 启动神经网络
network = Sequential()

# 添加有64个过滤器，一个大小为5x5的窗口和ReLU激活层函数的卷积层
network.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(channels, width, height), activation='relu'))

# 添加带一个2x2窗口的最大池化层
network.add(MaxPooling2D(pool_size=(2, 2)))

# 添加Dropout层
network.add(Dropout(0.5))

# 添加一个层来压平输入
network.add(Flatten())

# 添加带ReLU激活函数的有128个神经元的全连接层
network.add(Dense(128, activation="relu"))

# 添加Dropout层
network.add(Dropout(0.5))

# 添加使用softmax激活函数的全连接层
network.add(Dense(number_of_classes, activation="softmax"))

# 编译神经网络
network.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# 训练神经网络
network.fit(features_train, target_train, epochs=2, batch_size=1000, validation_data=(features_test, target_test))
