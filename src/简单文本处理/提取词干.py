from nltk.stem import PorterStemmer

# 创建单词序列
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# 创建词干转换器
porter = PorterStemmer()

# 应用词干转换器
print([porter.stem(word) for word in tokenized_words])
