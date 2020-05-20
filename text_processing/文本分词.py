from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# 创建文本
string = "Like we used to do"

# 分词
print(word_tokenize(string))

string = "We don't talk anymore. We don't talk anymore. We don't talk anymore. Like we used to do."

# 切分成句子
print(sent_tokenize(string))
