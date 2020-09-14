from nltk.corpus import stopwords

# 创建单词序列
tokenized_words = ['i', 'am', 'going', 'to', 'go', 'to', 'the', 'store', 'and', 'park']

# 暂停加载词
stop_words = stopwords.words('english')

# 删除停止词
print([word for word in tokenized_words if word not in stop_words])

# 查看停止词
print(stop_words[:5])
