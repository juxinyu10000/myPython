#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 车路协同实验添加设备
"""
import requests
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from zjhw.config import username
from zjhw.config import passwd
from zjhw.config import host


db_host = '123.60.33.112'
db_passwd = 'Zjhw@123'
database = 'mp_iot'

login_info = login(username, passwd)

header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

add_body = {
    "deviceName": "鞠新宇的小车3",
    "nodeId": "jxycxxc003",
    "factorySeq": "666005",
    "platformType": 2,
    "groupId": "959417018593849344",
    "deviceType": 4,  # 小车：4  摄像头：11 交通灯：7 沙盘：10
    "parentDeviceId": "3a231938-572e-43ee-8e30-eb08638806cb"
}

add_body1 = {  # 添加沙盘body
    "deviceName": "预埋数据沙盘1",
    "nodeId": "clxt123457",
    "factorySeq": "100002",
    "platformType": 2,
    "groupId": "959464186511896576",
    "deviceType": 10,
    "sceneType": 1
}


def add_testDev():
    """
    可添加沙盘/小车/摄像头等设备
    :return:
    """
    API = "/service-iot/api/v1/device/create"
    url = "https://%s:%d%s" % (host, API)
    resp = requests.post(url, json=add_body, headers=header)
    if "4000200" in resp.text:
        print(add_body['deviceName'] + "添加成功")
    else:
        print(f"出现了异常，{resp.text}")



def del_testDev(data):
    """
    删除沙盘/小车/摄像头等设备
    :param data: 被删除设别的ID
    :return:
    """
    API = "/service-iot-lab/api/v1/device/delete?" + data
    url = "" \
          "%s%s" % (host, API)
    print(url)
    resp = requests.delete(url, headers=header)
    if "4000200" in resp.text:
        print("删除成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 18
    if tag == 4:
        for i in range(30):
            a = str(i + 1).zfill(3)
            b = i + 1
            add_body["deviceName"] = "鞠新宇的CSXC" + str(b)
            add_body["nodeId"] = "jxyCSXC" + str(b)
            add_body["factorySeq"] = "410" + a
            add_body["deviceType"] = 4
            add_testDev()
    elif tag == 11:
        for i in range(2):
            a = str(i + 1).zfill(3)
            b = i + 1
            add_body["deviceName"] = "鞠新宇的CSSXT" + str(b)
            add_body["nodeId"] = "jxyCSSXT" + str(b)
            add_body["factorySeq"] = "110" + a
            add_body["deviceType"] = 11
            add_testDev()
    elif tag == 10:
        for i in range(20):
            a = str(i + 1).zfill(2)
            add_body1["deviceName"] = "预埋数据的沙盘" + str(i + 1)
            add_body1["nodeId"] = "ABCD" + str(i + 1)
            add_body1["factorySeq"] = "1020" + a
            add_testDev()

    else:
        # 下面代码是删除
        sql = "select device_id from device where device_name like '鸿蒙NB试验箱%'"
        date = mysql_connect(db_host, db_passwd, database, sql)
        for id in date:
            data1 = "deviceId=" + id[0]
            print(data1)
            del_testDev(data1)
