from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建随机森林分类器对象
random_forest = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)

# 训练模型
model = random_forest.fit(features, target)

# 查看袋外误差
print(random_forest.oob_score_)
