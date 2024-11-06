#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 算法管理添加
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, passwd

# token = login(username, passwd)
# header = {
#     "Content-Type": "application/json; charset=UTF-8",
#     "token": token}
body = {
    "description": "",
    "files": [{
        "fileName": "jxy",
        "moduleType": "develop",
        "dataType": "文件夹",
        "filePath": "data/develop/5703/jxy",
        "description": "",
        "updateTime": "2022-05-30 15:23:11",
        "source": "数据中心-开发任务",
        "userName": "juxinyu",
        "directory": "data/develop/5703",
        "userId": "5703",
        "version": "",
        "defaultStatus": "0",
        "dataId": "",
        "labelId": "",
        "createTime": "2022-05-30T05:54:16.214Z",
        "fileSize": "0.01MB",
        "width": "",
        "datasetId": "",
        "id": "J6OGE4EBtSsy4IEru8H5",
        "labelDate": "",
        "height": ""
    }],
    "source": "81",
    "dataType": "19",
    "scenarios": "25",
    "name": "Algorithm-test-001",
    "jobId": 121,
    "shareId": 0
}


def add_algorithms():
    API = "/service-dl-platform/api/v1/ams/algorithms"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(body['name'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    c = 1
    while c < 100:
        d = str(c).zfill(3)
        username = "juxinyu_student1" + d
        token = login(username, passwd)
        header = {
            "Content-Type": "application/json; charset=UTF-8",
            "token": token}
        c = c + 1
        tag = 1
        if tag == 1:
            for i in range(5):
                a = str(i + 1).zfill(5)
                body['name'] = "Algorithm-test-" + a
                add_algorithms()
