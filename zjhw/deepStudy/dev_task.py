#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加开发任务
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, passwd

# token = login(username, passwd)
# header = {
#     "Content-Type": "application/json; charset=UTF-8",
#     "token": token}


body = {
    "imageUrl": "harbor.dl.com/dl-library/jupyter-tf1.15-cpu-arm64:v0.0.1",
    "systemArch": "aarch64",
    "command": "dev_start.sh",
    "args": "jupyter-lab",
    "description": "",
    "dsPath": [],
    "image": "develop-tensorflow1.15-cpu-aarch64",
    "jobName": "Develop-0615-xingneng11",
    "outPath": "data/develop/5717/1da79",
    "resourceFlavorId": 41
}


def add_task():
    API = "/service-dl-training/api/v1/dpm/develop/"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(body['jobName'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


body1 = {
    "fileName": "jxy0616",
    "directory": "data/develop/5703",
    "moduleType": "develop",
    "dataType": "",
    "description": ""
}


def add_folder():
    API = "/service-dl-data/api/v1/ems/record/folder"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body1, headers=header)
    if "4000200" in resp.text:
        print(body1['fileName'] + "添加成功")
    else:
        print(f"出现了异常，{resp.json()}")

if __name__ == '__main__':
    c = 4
    while c < 50:
        d = str(c).zfill(3)
        username = "juxinyu_student2" + d
        result = login(username, passwd)
        header = {
            "Content-Type": "application/json; charset=UTF-8",
            "token": result[0]}
        tag = 1
        if tag == 1:
            body1['directory'] = "data/develop/" + str(result[1])
            #print(body1['directory'])
            add_folder()
            body['jobName'] = "Develop-0616-xingneng" + str(c)
            body['outPath'] = body1['directory'] + '/jxy0616'
            print(body['outPath'])
            print(body['jobName'])
            add_task()
            c = c + 1
