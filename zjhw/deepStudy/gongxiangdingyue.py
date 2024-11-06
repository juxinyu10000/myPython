#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加开发任务
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host


def dingyue():
    API1 = "/service-dl-platform/api/v1/kms/knowledge_square/ordering/11"
    API2 = "/service-dl-platform/api/v1/kms/knowledge_square/ordering/12"
    url1 = "%s%s" % (host, API1)
    url2 = "%s%s" % (host, API2)
    resp1 = requests.put(url1, headers=header)
    resp2 = requests.put(url2, headers=header)
    if "4000200" in resp1.text:
        print(username + "数据集订阅成功")
    else:
        print(f"出现了异常，{resp1.json()}")
    if "4000200" in resp2.text:
        print(username + "算法订阅成功")
    else:
        print(f"出现了异常，{resp1.json()}")


def add_dataSet():
    API = "/service-dl-platform/api/v1/dms/dataSet"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body1, headers=header)
    if "4000200" in resp.text:
        print(username + "数据集添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def add_suanfa():
    API = "/service-dl-platform/api/v1/ams/algorithms"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body2, headers=header)
    if "4000200" in resp.text:
        print(username + "算法添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def add_xunlian():
    API = "/service-dl-training/api/v1/dpm/training"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body3, headers=header)
    if "4000200" in resp.text:
        print(username + "训练任务添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def creatFolder():
    API = "/service-dl-data/api/v1/ems/record/folder"
    url = "%s%s" % (host, API)
    print(body4)
    resp = requests.post(url, json=body4, headers=header)
    if "4000200" in resp.text:
        print(username + "文件夹创建成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    c = 11
    folder = 5675
    while c < 50:
        username = "xueyuan%d" % c
        result = login(username, "a123456.")
        header = {
            "Content-Type": "application/json; charset=UTF-8",
            "token": result[0]}

        body1 = {
            "name": "Data-0912-" + username,
            "description": "",
            "industry": "",
            "dsPath": "",
            "dsType": "19",
            "source": "18",
            "scenarios": "25",
            "shareId": 11,
            "files": [{
                "fileName": "dataset-0911-gpu-test",
                "moduleType": "dataset",
                "dataType": "文件夹",
                "filePath": "data/knowledge_square/5663/dataset/11/dataset-0911-gpu-test",
                "description": "",
                "updateTime": "2023-09-12 10:05:54",
                "source": "共享数据集",
                "directory": "data/dataset/5663",
                "userId": "5663",
                "version": "V0001",
                "catalogStatus": "1",
                "dataId": "11",
                "fileSize": "0MB",
                "id": "JzB1hYoBBT9f2xc9xAAi",
                "businessType": "dataset",
                "fileId": "MyFDgIoBBT9f2xc9Aiwb"
            }]
        }
        # add_dataSet()

        body2 = {
            "description": "",
            "files": [{
                "fileName": "algorithm-0911-gpu",
                "moduleType": "algorithm",
                "dataType": "文件夹",
                "filePath": "data/knowledge_square/5663/algorithm/12/algorithm-0911-gpu",
                "description": "",
                "updateTime": "2023-09-12 10:06:07",
                "source": "共享算法",
                "directory": "data/algorithm/5663",
                "userId": "5663",
                "version": "V0001",
                "catalogStatus": "1",
                "dataId": "12",
                "fileSize": "33.14MB",
                "id": "wzB2hYoBBT9f2xc9AwAt",
                "businessType": "algorithm",
                "fileId": "9CFDgIoBBT9f2xc9ti1v"
            }],
            "source": "83",
            "dataType": "19",
            "scenarios": "25",
            "name": "Code-0912-" + username,
            "jobId": 0,
            "shareId": 12
        }
        # add_suanfa()

        body4 = {
            "fileName": username,
            "directory": "data/training/%d" % folder,
            "moduleType": "training",
            "dataType": "",
            "description": "训练任务新文件夹"
        }

        body3 = {
            "jobName": "Training-0912-" + username,
            "jobDescribe": "",
            "scenario": "25",
            "algId": 10,
            "algVersionId": 10,
            "modelId": "",
            "modelVersionId": "",
            "datasetId": 18,
            "datasetVersionId": 19,
            "validationDatasetId": "",
            "validationDatasetVersionId": "",
            "inputShape": "[28,28,1]",
            "trainingFrame": "pytorch",
            "trainingFrameVersion": "v1.5",
            "image": "pytorch1.13-cpu-x86_64",
            "imageUrl": "harbor.dl.com/dl-library/pytorch1.13-cpu-x86_64:v0.0.1",
            "jobType": 1,
            "jobNodeCount": 1,
            "resourceFlavorId": 8,
            "hyperparameters": "",
            "command": "train_start.sh",
            "args": "/data/codes/data/knowledge_square/5663/algorithm/12/algorithm-0911-gpu/distribution/paixu.py",
            "dataPath": ["data/knowledge_square/5663/dataset/11/dataset-0911-gpu-test",
                         "data/knowledge_square/5663/algorithm/12/algorithm-0911-gpu"],
            "outPath": "data/training/%d/" % folder + username,
            "isVisual": 0,
            "isTemplate": 0,
            "visualImage": "",
            "visualImageUrl": "",
            "systemArch": "x86_64"
        }
        body4['fileName'] = username
        creatFolder()
        add_xunlian()
        c = c + 1
        folder = folder + 1
