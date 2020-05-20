from nltk import pos_tag
from nltk import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer

# 创建文本
text_data = "Chris loved outdoor running"

# 使用预训练的词性标注器
text_tagged = pos_tag(word_tokenize(text_data))

# 查看词性
print(text_tagged)

'''
NNP：单数专有名词
NN：单数或复数的名词
RB：副词
VBD：过去式的动词
VBG：动名词或动词的现在分词形式
JJ：形容词
PRP：人称代词
'''
print([word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']])

# 创建文本
tweets = ["I am eating a burrito for breakfast", "Political science is an amazing field",
          "San Francisco is an amazing city"]

# 创建列表
tagged_tweets = []

# 为每条推文中的每个单词加标签
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# 使用ine-hot编码将标签扎UN哈UN成特征
one_hot_muti = MultiLabelBinarizer()
print(one_hot_muti.fit_transform(tagged_tweets))

# 使用classes_查看词名能发现每个特征都是一个词性标签
print(one_hot_muti.classes_)
