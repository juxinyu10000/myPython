#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 车路协同实验绑定设备
"""

import requests
from zjhw.lab_login import login
from zjhw.config import username
from zjhw.config import passwd
from zjhw.config import host

database = 'mp_iot'
labReportId="1062389626322362368"


def dev_bind(data):
    API = "/service-iot/api/v1/device/bind?" + data
    url = "%s%s" % (host, API)
    resp = requests.post(url, headers=header)
    if "4000200" in resp.text:
        print(username + "绑定成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    for i in range(100):
        username = "huawei" + str(i + 1)
        str_num = str(i + 1).zfill(2)
        login_info = login(username, passwd)
        header = {"Content-Type": "application/json",
                  "access_token": login_info['access_token'],
                  "application_key": login_info['application_key'],
                  "group_id": str(login_info['groupid'])}
        data = "parentDeviceId=61d6d60bde9933029be0bbec_20230104" + str_num + "&nodeId=20230111" + str_num + "&classId=959419979126550528&labReportId="+labReportId
        dev_bind(data)

    """
    #绑定交通灯
    for i in range (3):
        username = "student" + str(i + 1)
        login_info = login(username, passwd)
        header = {"Content-Type": "application/json",
                  "access_token": login_info['access_token'],
                  "application_key": login_info['application_key'],
                  "group_id": str(login_info['groupid'])}
        data = "23bda18c-19a6-4323-b154-b9232c3480d0&nodeId=jxycxjtdt00" + str(i + 1) + "&classId=950444464353742848&labReportId=956546842606030848"
        dev_bind(data)

    """
