#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 创建课程笔记
"""
import requests
from zjhw.lab_login import login
from config import host, db_host, db_passwd, db_user, passwd

shortName = "Huawei01"
passwd = 'Admin@123'

body = {
    "sectionId": "1031853010797797376",
    "courseId": "1031853010772631552",
    "content": "用户名—课程笔记",
    "classId": "1037659134765674496",
    "imageUrls": []
}


def createt_note():
    API = '/service-social/api/v1/note/create'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(username + "新增笔记成功:" + body["content"])
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(60):
            n = str(i + 1).zfill(2)
            username = shortName + n
            login_info = login(username, passwd)
            header = {"Content-Type": "application/json",
                      "access_token": login_info['access_token'],
                      "application_key": login_info['application_key'],
                      "group_id": str(login_info['groupid'])}

            for j in range(20):
                body["content"] = username + "课程笔记_" + str(i + 1) + "_" + str(j)
                createt_note()
