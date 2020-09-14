import numpy as np
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

np.random.seed(0)

number_of_features = 100

features, target = make_classification(n_samples=10000, n_features=number_of_features, n_informative=3, n_redundant=0,
                                       n_classes=2, weights=[.5, .5], random_state=0)


# 创建一个函数，返回编译过的神经网络
def create_network(optimizer="rmsprop"):
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features, )))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return network


# 封装Keras模型，以便它能被scikit-learn使用
neural_network = KerasClassifier(build_fn=create_network)

# 创建超参数空间
epochs = [5, 10]
batches = [5, 10, 100]
optimizers = ["rmsprop", "adam"]

# 创建超参数选项
hyper_parameters = dict(epochs=epochs, batch_size=batches, optimizer=optimizers)

# 创建网格搜索
grid = GridSearchCV(estimator=neural_network, param_grid=hyper_parameters)

# 实现网格搜索
grid_result = grid.fit(features, target)

# 查看最优神经网络的超参数
print(grid_result.best_params_)
