import os
import cv2
import math


def dataEnhance_fog(img_path):
    """
    :param img_path: 输入图片的路径，如：”\image\chuyu\shucai_1.jpg“
    :return:
    """

    img = cv2.imread(img_path)
    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.08  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    # cv2.imshow("fog", img_f)
    # cv2.waitKey()
    return img_f


if __name__ == '__main__':
    i = 1
    Output_dir = r".\image\qita\fog"  # 修改后图片保存路径
    image_path = r".\image\qita"
    picName = "qita_"
    for image in os.listdir(image_path):
        if image[-4:] == ".jpg":
                image_path1 = os.path.join(image_path, image)
                saveImg = os.path.join(Output_dir, picName + str(i) + ".jpg")
                i = i + 1
                fogimage = dataEnhance_fog(image_path1)
                # cv2.imshow("fog", fogimage)
                # cv2.waitKey()
                cv2.imwrite(saveImg, fogimage*255)
