import cv2 as cv
import ddddocr

class verificationCodeRecognition:

    def identify_gap(self, bp,tp):
        """
        缺口拖动验证码识别
        :param bp: 背景图片
        :param tp: 缺口图片
        :return:缺口图片左上角和右下角坐标
        """

        # 读取背景图片和缺口图片
        bg_img = cv.imread(bp) #背景图片
        tp_img = cv.imread(tp) #缺口图片

        # 识别图片边缘
        bg_edge = cv.Canny(bg_img, 100, 200)
        tp_edge = cv.Canny(tp_img, 100, 200)

        # 转换图片格式
        bg_pic = cv.cvtColor(bg_edge, cv.COLOR_GRAY2BGR)
        tp_pic = cv.cvtColor(tp_edge, cv.COLOR_GRAY2BGR)

        # 计算缺口位置
        result = cv.matchTemplate(bg_pic, tp_pic, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + tp_img.shape[1], top_left[1] + tp_img.shape[0])
        print("缺口左上角坐标：" + str(top_left))
        print("缺口右下角坐标：" + str(bottom_right))
        return top_left, bottom_right

    def Slider(self,bp,tp):
        """
        ddddocr库实现滑块验证码
        :param bp: 背景图片
        :param tp: 缺口图片
        :return: 缺口图片左上角和右下角坐标
        """
        det = ddddocr.DdddOcr()  # 实例化ocr对象
        with open(bp, 'rb') as f:  # 读取背景图片文件
            bg_bytes = f.read()
        with open(tp, 'rb') as f:  # 读取缺口图片文件
            tp_bytes = f.read()
        result = det.slide_match(bg_bytes,tp_bytes,simple_target=True)
        print(result)


    def textCode(self,img):
        """
        验证码识别
        :param img: 验证码图片
        :return: 验证码字符串
        """
        ocr = ddddocr.DdddOcr()  # 实例化ocr对象
        with open(img,'rb') as f:  # 读取图片文件
            img_bytes = f.read()
        result = ocr.classification(img_bytes) # 识别图片中验证码
        print(result)
        # 显示图片
        cv.imshow("code", cv.imread(img))
        cv.waitKey(0)
        cv.destroyAllWindows()
        return result


    def characterRecognition(self,img):
        """
        字符识别
        :param img: 验证码图片
        :return: 验证码字符串
        """
        ocr = ddddocr.DdddOcr(det=True)  # 实例化ocr对象
        with open(img,'rb') as f:  # 读取图片文件
            image = f.read()
        poses = ocr.detection(image)
        im = cv.imread(img)
        for box in poses:
            x1,y1,x2,y2 = box
            im1 = cv.rectangle(im,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)

        # 显示图片
        cv.imshow("code", im1)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    bg = "D:/myProject/myPytest/test_img/shadeImage2.png"
    tp = "D:/myProject/myPytest/test_img/cutoutImage2.png"
    codeImg = "D:/myProject/myPytest/test_img/yan7.png"
    charImg = "D:/myProject/myPytest/test_img/character1.png"
    vcr = verificationCodeRecognition()
    #vcr.identify_gap(bg, tp)
    #vcr.Slider(bg, tp)
    #vcr.textCode(codeImg)
    vcr.characterRecognition(charImg)