#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增教知识点
"""
import requests
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from config import host, db_host, db_passwd, db_user, username, passwd

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

add_body = {
    "parentId": "1116301115340443648",
    "name": "新建目录",
    "level": 2,
    "type": 2
}


def add_dir():
    API = '/service-lab/api/v1/catalog/index/create'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=add_body, headers=header)
    if "4000200" in resp.text:
        print(resp.json()['data']['name'] + "添加成功")
        # return resp.json()['data']['id']
    else:
        print(f"出现了异常，{resp.json()}")


mov_body = {
    "id": "1116308971078897664",
    "realName": "物联网实验001",
    "catalogIndexId": "1116301162773827584",
    "tags": [],
    "type": ""
}


def mov_cw(a):
    API = '/service-lab/api/v1/material/update'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=mov_body, headers=header)
    if "4000200" in resp.text:
        print(r'%d 课件移动成功' % a)
        # return resp.json()['data']['id']
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 2
    if tag == 1:
        for i in range(300):
            n = str(i).zfill(3)
            # print(n)
            add_body['name'] = n
            add_dir()
    elif tag == 2:
        sql_dir = "select id from catalog_index ci where ci.status =1 and ci.parent_id =1116301115340443648"
        date1 = mysql_connect(db_host, db_user, db_passwd, "mp_lab", sql_dir)
        sql_file = "select id,name  from material m where m.name like '物联网实验%'"
        date2 = mysql_connect(db_host, db_user, db_passwd, "mp_lab", sql_file)
        for i in range(300):
            mov_body['id'] = date2[i][0]
            mov_body['realName'] = date2[i][1]
            mov_body['catalogIndexId'] = date1[i][0]
            mov_cw(i)
