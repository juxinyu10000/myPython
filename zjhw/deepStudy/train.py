#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增训练任务
"""

import requests
from zjhw.deepStudy.dpstudyLogin import login
from zjhw.config import host, username, passwd

token = login(username, passwd)
header = {
    "Content-Type": "application/json; charset=UTF-8",
    "token": token}
body = {
    "jobName": "Training-0607-jxy",
    "jobDescribe": "",
    "scenario": "25",
    "algId": 533,
    "algVersionId": 545,
    "modelId": "",
    "modelVersionId": "",
    "datasetId": 713,
    "datasetVersionId": 719,
    "validationDatasetId": "",
    "validationDatasetVersionId": "",
    "inputShape": "[1024,1024,1]",
    "trainingFrame": "mindspore",
    "trainingFrameVersion": "v0.5",
    "image": "training-mindspore1.0-npu-aarch64",
    "imageUrl": "harbor.dl.com/dl-library/mindspore1.0-npu-arm64:v0.0.1",
    "jobType": 1,
    "jobNodeCount": 1,
    "resourceFlavorId": 22,
    "hyperparameters": "",
    "command": "train_start.sh",
    "args": "",
    "dataPath": ["data/dataset/logo", "data/develop/jxy"],
    "outPath": "data/training/jxy",
    "isVisual": 1,
    "isTemplate": 0,
    "visualImage": "mindinsight1.0-cpu-aarch64",
    "visualImageUrl": "harbor.dl.com/dl-library/mindinsight-arm64:v0.0.1",
    "systemArch": "aarch64"
}


def add_train_task():
    API = "/service-dl-training/api/v1/dpm/training"
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print(body['jobName'] + "添加成功")
        print(body)
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
      c = 1
      while c < 10:
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
                body['jobName'] = username + "-test1-" + a
                add_train_task()
