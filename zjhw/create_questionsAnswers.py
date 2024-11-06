#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 创建课程问答,包含创建问题方法和创建答案方法
"""
import requests
from zjhw.lab_login import login
from zjhw.config import host

shortName = "Huawei01"
passwd = 'Admin@123'

body1 = {
    "content": "规划局",
    "courseId": "1031853010772631552",
    "classId": "1037659134765674496",
    "sectionId": "1031853010797797376",
    "sectionName": "物联网实验",
    "coursewareId": "1022504515930427392",
    "coursewareName": "应用部署和发布-课件092201.pdf",
    "courseName": "测试URL实验课程-1018",
    "chapterId": "1031853010785214464",
    "chapterName": "物联网",
    "imageUrls": []
}


def create_question():
    API = '/service-social/api/v1/questions-answers/course/release'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body1, headers=header)
    if "4000200" in resp.text:
        print(username + "新增问答成功:" + body1["content"])
    else:
        print(f"出现了异常，{resp.json()}")


body2 = {
    "content": "回复用户名+回答描述+编号",
    "quizzer": "学生11",
    "parentId": "1038107519753015296",
    "classId": "955520897233121280",
    "courseId": "1019921781669044224",
    "sectionId": "1019921781681627137",
    "sectionName": "第一节",
    "coursewareId": "1019921781710987265",
    "coursewareName": "测试课件(1).pptx",
    "courseName": "鞠新宇的课程5",
    "chapterId": "1019921781681627136",
    "chapterName": "第一章"
}


def create_answers():
    API = '/service-social/api/v1/questions-answers/course/reply'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body2, headers=header)
    if "4000200" in resp.text:
        print(body2['quizzer'] + "的问题回答成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(60):
            n = str(i+1).zfill(2)
            username = shortName + n
            print (username,passwd)
            login_info = login(username, passwd)
            header = {"Content-Type": "application/json",
                      "access_token": login_info['access_token'],
                      "application_key": login_info['application_key'],
                      "group_id": str(login_info['groupid'])}
            for j in range(10):
                body1["content"] = body1["courseName"] + username + "如何查看某个容器资源使用情况_" + str(i) + "_" + str(j)
                create_question()
    if tag == 2:
        for i in range(1):
            username = shortName + str(i + 2)
            login_info = login(username, passwd)
            header = {"Content-Type": "application/json",
                      "access_token": login_info['access_token'],
                      "application_key": login_info['application_key'],
                      "group_id": str(login_info['groupid'])}

            body2['content'] = username + "的问题回复" + str(i)
            body2['quizzer'] = shortName + str(i + 1)
            body2['parentId'] = "1034757127736090624"
            create_answers()
