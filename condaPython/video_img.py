'''
读取本地视频，将视频切割成图片
'''

import shutil
import cv2 as cv
import os
import time

'''
playVideo：窗口展示本地视频，空格暂停，Esc退出播放
'''
def playVideo(videoPath):
    cap = cv.VideoCapture(videoPath)
    fps = cap.get(cv.CAP_PROP_FPS)
    count = cap.get(cv.CAP_PROP_FRAME_COUNT) #获取视频帧数
    print(fps, '\n', count)
    while cap.isOpened():
        # isOpened()检测视频初始化是否成功，成功返回True，否则返回False
        time.sleep(0.01)
        # ret为返回的布尔值，frame为返回的每一帧，如果没有帧，则返回None
        ret, frame = cap.read()
        # 设置窗口大小
        cv.namedWindow('video', 0)
        cv.resizeWindow('video', 420, 300)
        if ret:
            cv.imshow('video', frame)  # 展示视频流
        else:
            if frame is None:
                print("视频拉取失败，请检车摄像机网络")
                break
        key = cv.waitKey(10)  # 窗口图像刷新时间1毫秒
        if key == 32:  # 空格
            cv.imwrite('D:/myPython/condaPython/test_picture/pause_img.jpg', frame)
            cv.waitKey(0)  # 不刷新图像，实现暂停
            continue  # 再按继续
        if key == 27:  # Esc
            cv.imwrite('D:/myPython/condaPython/test_picture/break_img.jpg', frame)
            break
    cap.release()


# 将视频分割成图片
def video2image(videoPath):
    cap = cv.VideoCapture(videoPath)
    path = 'D:/myPython/condaPython/save_picture/'
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
    j = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            picture_name = '%ssavePicture_%d.jpg' % (path, j)
            cv.imwrite(picture_name, frame)
            j = j + 1
        else:
            break
    cap.release()


video_path = 'D:/myPython/condaPython/video/手势.mp4'
playVideo(video_path)
video2image(video_path)
