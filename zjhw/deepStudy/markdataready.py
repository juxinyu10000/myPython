#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加用户
"""
from random import choice
import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, passwd
from zjhw.DB_connect import mysql_connect
from zjhw.config import db_host, db_passwd, database, db_user


def dingyue():
    API = "/service-dl-platform/api/v1/kms/knowledge_square/ordering/112"
    url = "%s%s" % (host, API)
    resp = requests.put(url, headers=header)
    if "4000200" in resp.text:
        print("订阅成功")
    else:
        print(f"出现了异常，{resp.json()}")


body1 = {"labelType": 1, "name": "juxinyu_student2002"}


def add_createdir():
    API = "/service-dl-data/api/v1/label/dataset/create"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body1, headers=header)
    if "4000200" in resp.text:
        print("添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


body3 = {"pageNum": 1, "pageSize": 10, "queryRange": "mine"}


def query():
    API = "/service-dl-data/api/v1/label/dataset/list"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body3, headers=header)
    if "4000200" in resp.text:
        a = resp.json()["data"]["list"][0]["id"]
        print(a)
        return a
    else:
        print(f"出现了异常，{resp.json()}")


body2 = {"datasetId": 102, "haveLabel": 0, "labelType": 1, "dirId": "RkyyR4EBtSsy4IErodkp",
         "sourceDirectory": "data/knowledge_square/5703/dataset/112/towty"}


def importdata():
    API = "/service-dl-data/api/v1/label/dataset/import"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body2, headers=header)
    if "4000200" in resp.text:
        print("导入成功")
    else:
        print(f"出现了异常，{resp.json()}")


body5 = {"datasetId":113,"name":"鱼"}
def add_tag():
    API = "/service-dl-data/api/v1/label/create"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body5, headers=header)
    if "4000200" in resp.text:
        print(body5["name"] + "添加成功")
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

            ids = query()
            body5["datasetId"] = ids
            add_tag()
