#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加数据源
"""
from random import choice
import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, passwd
from zjhw.DB_connect import mysql_connect
from zjhw.config import db_host, db_passwd, database, db_user

body1 = {
    "ip": "192.168.88.145",
    "port": "3306",
    "loginUserName": "root",
    "loginPassword": "ZJHW_dl-dlvr@800"
}


def add_datebase():
    API = "/service-dl-data/api/v1/access/data/origin/add"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body1, headers=header)
    if "4000200" in resp.text:
        print("数据库连接成功")
        return resp.json()["data"]["dataOriginId"]
    else:
        print(f"出现了异常，{resp.json()}")


body2 = {
    "dataOriginId": 157,
    "dataOriginName": "174",
    "describe": ""
}


def updateorigin():
    API = "/service-dl-data/api/v1/access/data/origin/update"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body2, headers=header)
    if "4000200" in resp.text:
        print("数据源保存成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(100):
            b = str(i + 2).zfill(3)
            username = "juxinyu_student2" + b
            token = login(username, passwd)
            header = {
                # "Content-Type": "application/json; charset=UTF-8",
                "token": token}
            dataOriginId = add_datebase()
            body2["dataOriginId"] = dataOriginId
            updateorigin()
