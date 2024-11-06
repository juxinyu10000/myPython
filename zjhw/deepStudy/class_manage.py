#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加班级
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, username, passwd

token = login(username, passwd)
header = {
    "Content-Type": "application/json; charset=UTF-8",
    "token": token}
body = {
    "ids": "5647",
    "className": "计算机1班",
    "orgId": 187,
    "classOrder": 2,
    "classId": "",
    "validTime": ""
}


def add_class():
    API = "/service-user/api/user/saveClassUsers"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(body['className'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(9):
            body['className'] = "计算机" + str(i+2) + "班"
            add_class()
