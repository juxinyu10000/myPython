#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 创建签到表
"""
import requests
from zjhw.lab_login import login
from config import host, db_host, db_passwd, db_user, passwd

shortName = "Huawei01"
passwd = 'Admin@123'



def createt_sign(classid):
    API = '/service-edumanager/web/sign/create?classId=' + str(classid)
    url = "%s%s" % (host, API)
    resp = requests.post(url, headers=header)
    if "4000200" in resp.text:
        print("签到表新增成功:")
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


body = {
    "id": "1039579171792027648",
    "signName": "我的签到表"
}


def edi_sign():
    API = '/service-edumanager/web/sign/edit'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print("签到表修改成功:" + body["signName"])

    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    for i in range(60):
        n = str(i + 1).zfill(2)
        username = shortName + n
        login_info = login(username, passwd)
        header = {"Content-Type": "application/json",
                  "access_token": login_info['access_token'],
                  "application_key": login_info['application_key'],
                  "group_id": str(login_info['groupid'])}
        for j in range(50):
            signID = createt_sign(login_info['classId'])
            body["id"] = str(signID)
            body["signName"] = username + "的签到表" + str(j+1)
            edi_sign()
