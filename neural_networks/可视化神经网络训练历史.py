import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models, layers

# 设定随机种子
from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(0)

# 设定想要的特征数量
number_of_features = 10000

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

# 训练神经网络
history = network.fit(features_train, target_train, epochs=15, verbose=0, batch_size=1000,
                      validation_data=(features_test, target_test))

with PdfPages("神经网络训练历史可视化图.pdf") as pdf:
    # 获取训练集和测试集的损失历史数值
    training_loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    # 为每个epoch创建编号
    epoch_count = range(1, len(training_loss) + 1)
    # 画出损失的历史数值
    fig = plt.figure()
    plt.plot(epoch_count, training_loss, "r--")
    plt.plot(epoch_count, test_loss, "b-")
    plt.legend(["Training Loss", "Test Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Page1")
    pdf.savefig(fig)
    plt.close(fig)
    plt.show()
    # 获取训练集和测试集数据的准确率历史数值
    fig = plt.figure()
    training_accuracy = history.history["accuracy"]
    test_accuracy = history.history["val_accuracy"]
    plt.plot(epoch_count, training_accuracy, "r--")
    plt.plot(epoch_count, test_accuracy, "b-")
    # 可视化准确率的历史数值
    plt.legend(["Training Accuracy", "Test Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy Score")
    plt.title("Page2")
    pdf.savefig(fig)
    plt.close(fig)
    plt.show()

