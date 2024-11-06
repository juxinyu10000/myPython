#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增数据集列表
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, username, passwd

token = login(username, passwd)
header = {
    "Content-Type": "application/json; charset=UTF-8",
    "token": token}
body = {
    "name": "Data-0607-0d360b",
    "description": "",
    "industry": "3",
    "dsPath": "",
    "dsType": "19",
    "source": "14",
    "scenarios": "25",
    "shareId": "",
    "files": [{
        "fileName": "logo",
        "moduleType": "dataset",
        "dataType": "文件夹",
        "filePath": "data/dataset/7453/logo",
        "description": "",
        "updateTime": "2022-06-07 14:56:29",
        "source": "数据中心-数据集",
        "userName": "juxinyu",
        "directory": "data/dataset/7453",
        "userId": "7453",
        "version": "",
        "defaultStatus": "0",
        "dataId": "",
        "labelId": "",
        "createTime": "2022-06-07T06:56:16.172Z",
        "fileSize": "0.08MB",
        "datasetId": "",
        "id": "VZf0PIEBAxqRtJ2hIUV4",
        "labelDate": ""
    }]
}


def add_dataset():
    API = "/service-dl-platform/api/v1/dms/dataSet"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(body['name'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(495):
            a = str(i + 1).zfill(5)
            body['name'] = username + "-test-" + a
            add_dataset()
