import cv2

'''
使用OpenCV的图像切分代码
图像集下载地址：https://cs.nyu.edu/~roweis/data/olivettifaces.gif
cv2.imread()不支持gif，使用了图像转换工具：http://pic.55.la
'''
# 获取图像 gif不行
data = cv2.imread("olivettifaces.jpg")
# 转换为灰度图像
data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
# 将人脸图片提取为{label:list}形式
faces = {}
label = 0
count = 1
pic_list = []
for row in range(20):
    for column in range(20):
        pic_list.append(data[row*57:(row+1)*57, column*47:(column+1)*47])
        if count % 10 == 0 and count != 0:
            faces[label] = pic_list
            label += 1
            # 初始化一个新的列表
            pic_list = []
        count += 1
