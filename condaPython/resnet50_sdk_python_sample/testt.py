import random

import numpy as np  # 用于对多维数组进行计算
import cv2  # 图片处理三方库，用于对图片进行前后处理

pic_path = 'data/dog1.png'  # 单张图片

my_list = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
           [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]  # 3x3列表
arr = np.array(my_list)  # 列表转换成数组
# print(a[0,1,2])  #取数组某个值
# print(arr[:,:,1])

img_bgr = cv2.imread('data/test.jpg')
#img_bgr[:, :, 0] = 0  #使B（bluss）通道所有灰度值为0
#img_bgr[:, :, 1] = 0 #使G(Green)通道所有灰度值为0
#img_bgr[:, :, 2] = 0 #使R(Red)通道所有灰度值为0
img_rgb = img_bgr[:,:,::-1]

cv2.imshow("pic",img_rgb)
cv2.waitKey(0)
