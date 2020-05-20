import re

# 创建文本
text_data = ["    Loving him is like driving a new Maserati down a dead end street.    ",
             "Faster than the wind, passionate as sin, ending so suddenly.",
             "    Loving him is like trying change to your mind once you're already flying through the free fall.  "]

# 去除文本两端的空格
strip_whitespace = [string.strip() for string in text_data]

# 查看文本
print(strip_whitespace)

# 删除句点
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# 查看文本
print(remove_periods)


# 创建函数
def capitalizer(string: str) -> str:
    return string.upper()


# 应用函数
iter1 = iter(capitalizer(string) for string in remove_periods)
for i in iter1:
    print(i, end="\n")


# 创建正则表达式函数
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]]", "X", string)


# 应用函数
iter2 = iter(replace_letters_with_X(string) for string in remove_periods)
for i in iter2:
    print(i, end="\n")
