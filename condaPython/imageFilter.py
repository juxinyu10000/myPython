from PIL import Image
from PIL import ImageFilter
import os

Input_dir = r".\test_picture"  # 原图片路径
Output_dir = r".\save_picture\imageEnhance"  # 修改后图片保存路径

for name in os.listdir(Input_dir):
    img_path = os.path.join(Input_dir, name)
    img = Image.open(img_path)
    # img01 = img.filter(ImageFilter.BLUR)  # 增加模糊滤镜
    # img01.show()  # 展示图片
    img02 = img.filter(ImageFilter.CONTOUR)  # 提取图像轮廓
    img02.show()  # 展示图片
