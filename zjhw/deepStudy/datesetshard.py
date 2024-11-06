#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 数据集版本共享
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, username, passwd
from zjhw.config import db_host,db_user,db_passwd,database
from zjhw.DB_connect import mysql_connect

token = login(username, passwd)
header = {
    "Content-Type": "application/json; charset=UTF-8",
    "token": token}


def add_dataset(id,ds_id):
    API = "/service-dl-platform/api/v1/dms/dataSet/" + str(ds_id) + "/versions/" + str(id) + "/shared"
    url = "%s%s" % (host, API)
    resp = requests.put(url, headers=header)
    if "4000200" in resp.text:
        print(str(id) + "共享成功")
    else:
        print(f"出现了异常，{resp.json()}")

if __name__ == '__main__':
    tag = 1
    if tag == 1:
        sql = "select id,ds_id from ds_version where user_id=7453"
        date = mysql_connect(db_host, db_user, db_passwd, database, sql)
        for i in date:
            add_dataset(i[0], i[1])