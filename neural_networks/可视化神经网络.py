from keras import models, layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# 启动神经网络
network = models.Sequential()

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu", input_shape=(10, )))

# 添加使用ReLU激活函数的全连接层
network.add(layers.Dense(units=16, activation="relu"))

# 添加使用sigmoid激活函数的全连接层
network.add(layers.Dense(units=1, activation="sigmoid"))

# 可视化神经网络结构
SVG(model_to_dot(network, show_shapes=True).create(prog="dot", format="svg"))

# 将可视化后的网络结构图保存为文件
plot_model(network, show_shapes=True, to_file="network_normal.png")

# 下面展示一个更简单的神经网络
SVG(model_to_dot(network, show_shapes=False).create(prog="dot", format="svg"))
plot_model(network, show_shapes=False, to_file="network_simplicity.png")
