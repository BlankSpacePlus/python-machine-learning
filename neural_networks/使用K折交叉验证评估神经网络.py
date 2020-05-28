import numpy as np
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

np.random.seed(0)

number_of_features = 100

features, target = make_classification(n_samples=10000, n_features=number_of_features, n_informative=3, n_redundant=0,
                                       n_classes=2, weights=[.5, .5], random_state=0)


# 创建一个函数，返回编译过的神经网络
def create_network():
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features, )))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return network


# 封装Keras模型，以便它能被scikit-learn使用
neural_network = KerasClassifier(build_fn=create_network, epochs=10, batch_size=100)

# 使用3折交叉验证来评估神经网络
print(cross_val_score(neural_network, features, target, cv=3))
