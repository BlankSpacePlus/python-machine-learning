import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models, layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

np.random.seed(0)

features, target = make_regression(n_samples=10000, n_features=3, n_informative=3, n_targets=1, noise=0.0,
                                   random_state=0)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33,
                                                                            random_state=0)

network = models.Sequential()

network.add(layers.Dense(units=32, activation="relu", input_shape=(features_train.shape[1], )))

network.add(layers.Dense(units=32, activation="relu"))

network.add(layers.Dense(units=1))

network.compile(loss="mse", optimizer="RMSprop", metrics=["mse"])

# 训练神经网络
history = network.fit(features_train, target_train, epochs=10, verbose=0, batch_size=100,
                      validation_data=(features_test, target_test))
