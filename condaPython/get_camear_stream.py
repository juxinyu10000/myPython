"""
@author:juxinyu
@data:20230915
"""
import time

import cv2 as cv
import datetime


def current_time(fmt=None):
    if fmt is None:
        fmt = '%Y_%m_%d %H:%M:%S'
    return datetime.datetime.today().strftime(fmt)


# 摄像头RTSP 地址
camear_url = 'rtsp://admin:HuaWei123@192.168.10.110/LiveMedia/ch1/Media1'
# 获取网络摄像头视频，返回本机第一个摄像头视频填0
video_path = 'C:/Users/鞠新宇/Desktop/测试课件/AVI视频.avi'
cap = cv.VideoCapture(video_path)

while True:
    # ret为返回的布尔值，frame为返回的每一帧，如果没有帧，则返回None
    ret, frame = cap.read()
    cv.namedWindow('video',0)
    cv.resizeWindow('video',300,260)
    if ret:
        cv.imshow('video', frame)  # 展示视频流
    else:
        if frame is None:
            print("视频拉取失败，请检车摄像机网络")
            break
    key = cv.waitKey(1)  # 窗口图像刷新时间1毫秒
    if key == 32:  # 按下空格键停止
        break

cap.release()  # 关闭摄像头
