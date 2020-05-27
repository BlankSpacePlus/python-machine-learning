from sklearn.metrics import make_scorer, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

features, target = make_regression(n_samples=100, n_features=3, random_state=1)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.10,
                                                                            random_state=1)


# 创建自定义指标函数
def custom_metric(target_test, target_predicted):
    # 计算R方得分
    r2 = r2_score(target_test, target_predicted)
    # 返回R方得分
    return r2


# 创建评分函数（评分器），并且定义分数越高代表模型越好
score = make_scorer(custom_metric, greater_is_better=True)

# 创建岭回归对象
classifier = Ridge()

# 训练岭回归模型
model = classifier.fit(features_train, target_train)

# 定义自定义评分器
print(score(model, features_test, target_test))

# 对测试集进行评测
target_predicted = model.predict(features_test)

# 计算R方得分
print(r2_score(target_test, target_predicted))
