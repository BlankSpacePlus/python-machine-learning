import unicodedata
import sys

# 创建文本
text_data = ['Hi!!!!!I. Love. This  Song....,;;', '100000  Agree%%&*%!!! #LoveI T', 'Right?!?!?!']

# 创建字典
punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

# 移除每个字符串中的标点
print([string.translate(punctuation) for string in text_data])
