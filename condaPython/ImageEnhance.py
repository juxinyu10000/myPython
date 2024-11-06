from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
import os

'''
函 数 名：contrastEnhancement(root_path, img_name, contrast)
函数功能：对比度增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        contrast ：对比度
返 回 值：
        对比度增强后的图片
'''


def contrastEnhancement(root_path, image_name, contrast):
    image = Image.open(os.path.join(root_path, image_name))
    enh_con = ImageEnhance.Contrast(image)  # 增强图像对比度
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


'''
函 数 名：brightnessEnhancement(root_path,img_name,brightness)
函数功能：亮度增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        brightness ：亮度
返 回 值：
        亮度增强后的图片
'''


def brightnessEnhancement(root_path, img_name, brightness):
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)  # 增强图片亮度
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


'''
函 数 名：colorEnhancement(root_path,img_name,color)
函数功能：颜色增强
入口参数：
        root_path ：图片根目录
        img_name ：图片名称
        color ：颜色
返 回 值：
        颜色增强后的图片
'''


def colorEnhancement(root_path, img_name, color):
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    image_colored = enh_col.enhance(color)
    return image_colored


'''
函 数 名：gaussian_noise(img, mean, sigma)
函数功能：添加高斯噪声
入口参数：
        img ：原图
        mean ：均值
        sigma ：标准差
返 回 值：
        噪声处理后的图片
'''


def gaussian_noise(img, mean, sigma):
    # 将图片灰度标准化
    img = (img / 255)
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 这里也会返回噪声，注意返回值
    return gaussian_out


'''
函 数 名：motion_blur(image)
函数功能：运动模糊化处理
入口参数：
        image ：原图
返 回 值：
        模糊化处理后的图片
'''


def motion_blur(image):
    degree = 8
    angle = 45
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


'''
函数功能：将原图进行高斯和模糊化变换
入口参数：
        input_dir ：原图路径
        filename ：原图名称
返 回 值：
        模糊处理后和噪声处理后的图片
'''


def gauss_fuzzy_convert(input_dir, filename):
    path = input_dir + "/" + filename  # 获取文件路径
    fuzzy_image = cv2.imread(path)  # 读取图片
    noise_img = cv2.imread(path)  # 读取图片
    fuzzy_image_1 = motion_blur(fuzzy_image)  # 模糊化
    noise_img_1 = gaussian_noise(noise_img, 0.1, 0.08)  # 高斯噪声
    # cv2.imshow('rain_effct', fuzzy_image_1)
    # cv2.waitKey()
    # cv2.destroyWindow('rain_effct')
    return fuzzy_image_1, noise_img_1


Input_dir = r".\image\qita"
n = 1
for name in os.listdir(Input_dir):
    print(name)
    if name[-4:] == ".jpg":
        for i in [0.7, 1.2]:
            newName = name[:-4] + "_" + str(i) + ".jpg"

            bright_path = Input_dir + "/bright"
            saveImage = brightnessEnhancement(Input_dir, name, i)  # 图像亮度增强0.5倍，1.5倍
            saveImage.save(os.path.join(bright_path, "bright" + newName))  # 效果增强后保存

            contrast_path = Input_dir + "/contrast"
            saveImage = contrastEnhancement(Input_dir, name, i)  # 对比度增强0.5倍，1.5倍
            saveImage.save(os.path.join(contrast_path, "contrast" + newName))

            color_path = Input_dir + "/color"
            saveImage = colorEnhancement(Input_dir, name, i)  # 图像颜色增强0.5倍，1.5倍
            saveImage.save(os.path.join(color_path, "color" + newName))

        noise_path = Input_dir + r"/noise"
        fuzzy_img, noise_img = gauss_fuzzy_convert(Input_dir, name)
        cv2.imwrite(os.path.join(noise_path, " fuzzy" + newName), fuzzy_img)  # 保存模糊处理后的图片
        cv2.imwrite(os.path.join(noise_path, " noise" + newName), noise_img)  # 保存噪声处理后的图片
