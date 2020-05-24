from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建standardizer
standardizer = StandardScaler()

# 标准化特征
features_standardized = standardizer.fit_transform(features)

# 创建一个KNN分类器
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# 创建一个流水线
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

# 确认一个流水线
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

# 创建grid搜索
# classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# classifier.fit(features_standardized, target)

# 显示最佳邻域的大小（k值）
# print(classifier.estimator.get_params()["knn__n_neighbors"])

'''
BUG 真值Wie6 求得为5
'''