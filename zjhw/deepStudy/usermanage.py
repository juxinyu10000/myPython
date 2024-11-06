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
from zjhw.config import host, username, passwd
from zjhw.DB_connect import mysql_connect
from zjhw.config import db_host,db_passwd,database,db_user

token = login(username, passwd)
header = {
    #"Content-Type": "application/json; charset=UTF-8",
    "token": token}
payload = {'name': 'juxinyu_student2',
           'realName': '王大锤',
           'pwd': '76b566d90f43b5097261febbef7cab77',
           'phone': '13111111111',
           'sex': '男',
           'number': '100001',
           'roleIds': '80',
           'institutionId': '187',
           'ids': '19'}


def add_user():
    API = "/service-user/api/user/regist"
    url = "%s%s" % (host, API)
    print (url)
    resp = requests.post(url, data=payload, headers=header)
    if "4000200" in resp.text:
        print(payload['name'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        sql = "select id from class where class_name like '计算机%'"
        date = mysql_connect(db_host,db_user, db_passwd, database, sql)
        b = 2
        for j in date:
            print (j[0])
            for i in range(200):
                    a = str(b) + str(i + 2).zfill(3)
                    payload['name'] = 'juxinyu_student' + a
                    payload['realName'] = choice(('彭于晏','柳岩'))
                    payload['phone'] = '1311111' + a
                    payload['sex'] = choice(('男','女'))
                    payload['number'] = '10' + a
                    payload['ids'] = j[0]
                    add_user()
            b = b + 1

