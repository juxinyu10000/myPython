#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增考题
"""
import requests
from zjhw.config import host
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from config import host, db_host, db_passwd, db_user, username, passwd

username = username
passwd = passwd

login_info = login(username, passwd)
header = {"Content-Type": "application/json;charset=UTF-8",
          "token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}
body = {
    "answer": 2,
    "complexity": "0",
    "description": "",
    "options": [{
        "option": "<p>i am fine</p>",
        "code": 0,
        "type": 0,
        "key": 1
    }, {
        "option": "<p>i am juxinyu</p>",
        "code": 1,
        "type": 0,
        "key": 2
    }, {
        "option": "<p>how are you</p>",
        "code": 2,
        "type": 0,
        "key": 3
    }, {
        "option": "<p>right</p>",
        "code": 3,
        "type": 0,
        "key": 4
    }],
    "stem": "<p>what's your name?</p>",
    "type": "0",
    "knowledgePointId": 237
}


def add_question():
    API = '/service-exam/question/v2.0/save'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print("试题创建成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(4000):
            add_question()