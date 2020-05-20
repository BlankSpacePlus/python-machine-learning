from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

# 从布朗语料库中获取文本数据，切分成句子
sentences = brown.tagged_sents(categories='news')

# 将4000个句子用作训练，623个句子用作测试
train = sentences[:4000]
test = sentences[4000:]

# 创建回退标注器
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff=unigram)
trigram = TrigramTagger(train, backoff=bigram)

# 查看准确率
print(trigram.evaluate(test))
