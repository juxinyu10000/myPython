import cv2 as cv
import time

# 读取图像文件路径
font = cv.FONT_HERSHEY_SIMPLEX  # 定义字体
text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
img = cv.imread(r'D:\myPython\image\dog.jpg', cv.IMREAD_COLOR)  # 读取本地图片，返回列表
img1 = cv.flip(img, 1)  # 翻转图片
img2 = cv.putText(img1, text, (5, 20), font, 0.5, (255, 0, 255))  # 图片上叠加文字
cv.imshow('image', img1)  # 窗口显示图片
# cv.imwrite(r'D:\Ul\huigou.jpg', img1)  #指定路径保存图片
cv.waitKey(0)
cv.destroyAllWindows()