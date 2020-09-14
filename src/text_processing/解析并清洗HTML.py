from bs4 import BeautifulSoup

string = ""

with open('test.html', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for i in lines:
        string = '%s %s' % (string, i)

# 解析HTML
soup = BeautifulSoup(string, "lxml")

# 查找id为"b_id"的div标签，并查看文本
print(soup.find("div", {"id": "b_id"}).text)
