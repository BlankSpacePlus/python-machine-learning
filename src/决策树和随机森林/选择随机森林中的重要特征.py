from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建随机森林分类器对象
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# 创建对象，选择重要性大于或等于阈值的特征
selector = SelectFromModel(random_forest, threshold=0.3)

# 使用选择器创建新的特征矩阵
features_important = selector.fit_transform(features, target)

# 使用重要特征训练随机森林模型
model = random_forest.fit(features_important, target)
