from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 加载手写数字的数据集
digits = datasets.load_digits()

# 创建特征矩阵
features = digits.data

# 创建目标向量
target = digits.target

# 创建标准化对象
standardizer = StandardScaler()

# 创建逻辑回归对象
logit = LogisticRegression(max_iter=10000)

# 创建包含数据标准化和逻辑回归的流水线
pipeline = make_pipeline(standardizer, logit)

# 创建k折交叉验证对象
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# 执行k折交叉验证 n_job=-1
cv_results = cross_val_score(pipeline, features, target, cv=kf, scoring="accuracy")

# 计算得分的平均值
print(cv_results.mean())

# 查看10份数据的得分
print(cv_results)

# 创建训练集和测试集
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1,
                                                                            random_state=1)

# 使用训练集计算标准化参数
standardizer.fit(features_train)

# 将标准化操作应用到训练集和测试集
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

# 创建一个流水线
pipeline = make_pipeline(standardizer, logit)

# 执行K交叉验证 n_job=-1
cv_results = cross_val_score(pipeline, features, target, cv=kf, scoring="accuracy")
