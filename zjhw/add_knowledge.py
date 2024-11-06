#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增教知识点
"""
import requests
from zjhw.config import host
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from config import host,db_host,db_passwd,db_user,username,passwd

database = 'mp_aggregation'
username = username
passwd = passwd

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

add_body = {
    "parentId": "",
    "knowledgeName": "新建节点"
}


def add_knowledge_catalogue():
    API = '/service-aggregation/api/v1/knowledge/catalog/add'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=add_body, headers=header)
    if "4000200" in resp.text:
        print(resp.json()['data']['knowledgeName'] + "添加成功")
        return resp.json()['data']['id']
    else:
        print(f"出现了异常，{resp.json()}")




def add_knowledge(threeid):
    API = '/service-aggregation/api/v1/knowledge/batch/add'
    url = "%s%s" % (host, API)
    add_body1 = {
        "batchAddKnowledge": [{
            "knowledgeId": threeid,
            "childKnowledgeList": [{
                "knowledgeId": 1648719178998,
                "knowledgeName": "1级知识点",
                "level": 3,
                "parentId": threeid,
                "sort": 1
            }, {
                "knowledgeId": 1648719181943,
                "knowledgeName": "2级知识点",
                "level": 4,
                "parentId": 1648719178998,
                "sort": 1
            }, {
                "knowledgeId": 1648719188833,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719181943,
                "sort": 1
            }, {
                "knowledgeId": 1648719203184,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719188833,
                "sort": 1
            }, {
                "knowledgeId": 1648719208810,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719188833,
                "sort": 2
            }, {
                "knowledgeId": 1648719190219,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719181943,
                "sort": 2
            }, {
                "knowledgeId": 1648719210523,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719190219,
                "sort": 1
            }, {
                "knowledgeId": 1648719212069,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719190219,
                "sort": 2
            }, {
                "knowledgeId": 1648719183711,
                "knowledgeName": "2级知识点",
                "level": 4,
                "parentId": 1648719178998,
                "sort": 2
            }, {
                "knowledgeId": 1648719191676,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719183711,
                "sort": 1
            }, {
                "knowledgeId": 1648719213408,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719191676,
                "sort": 1
            }, {
                "knowledgeId": 1648719214577,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719191676,
                "sort": 2
            }, {
                "knowledgeId": 1648719192879,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719183711,
                "sort": 2
            }, {
                "knowledgeId": 1648719215891,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719192879,
                "sort": 1
            }, {
                "knowledgeId": 1648719217285,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719192879,
                "sort": 2
            }, {
                "knowledgeId": 1648719180651,
                "knowledgeName": "1级知识点",
                "level": 3,
                "parentId": threeid,
                "sort": 2
            }, {
                "knowledgeId": 1648719185170,
                "knowledgeName": "2级知识点",
                "level": 4,
                "parentId": 1648719180651,
                "sort": 1
            }, {
                "knowledgeId": 1648719197183,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719185170,
                "sort": 1
            }, {
                "knowledgeId": 1648719218839,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719197183,
                "sort": 1
            }, {
                "knowledgeId": 1648719220522,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719197183,
                "sort": 2
            }, {
                "knowledgeId": 1648719198430,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719185170,
                "sort": 2
            }, {
                "knowledgeId": 1648719222647,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719198430,
                "sort": 1
            }, {
                "knowledgeId": 1648719224312,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719198430,
                "sort": 2
            }, {
                "knowledgeId": 1648719187558,
                "knowledgeName": "2级知识点",
                "level": 4,
                "parentId": 1648719180651,
                "sort": 2
            }, {
                "knowledgeId": 1648719199805,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719187558,
                "sort": 1
            }, {
                "knowledgeId": 1648719225930,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719199805,
                "sort": 1
            }, {
                "knowledgeId": 1648719227105,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719199805,
                "sort": 2
            }, {
                "knowledgeId": 1648719200932,
                "knowledgeName": "3级知识点",
                "level": 5,
                "parentId": 1648719187558,
                "sort": 2
            }, {
                "knowledgeId": 1648719228867,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719200932,
                "sort": 1
            }, {
                "knowledgeId": 1648719229948,
                "knowledgeName": "4级知识点",
                "level": 6,
                "parentId": 1648719200932,
                "sort": 2
            }]
        }],
        "batchDeleteKnowledgeId": []
    }
    resp = requests.post(url, json=add_body1, headers=header)
    if "4000200" in resp.text:
        print("知识点添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def del_knowledge(id):
    API = '/service-aggregation/api/v1/knowledge/can/delete?knowledgeId=' + id
    url = "%s%s" % (host, API)
    resp = requests.post(url, headers=header)
    if "4000200" in resp.text:
        print("删除成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 2
    if tag == 1:
        for i in range(5):
            add_body["knowledgeName"] = "一级知识点" + str(i + 1)
            add_body["parentId"] = ""
            knowledgeID1 = add_knowledge_catalogue()
            for j in range(5):
                add_body["knowledgeName"] = "二级知识点" + str(j + 1)
                add_body["parentId"] = knowledgeID1
                knowledgeID2=add_knowledge_catalogue()
                add_knowledge(knowledgeID2)



    if tag == 2:
        sql = "select id from knowledge_tree where create_user ='" + username + "' and `level`=3"
        date = mysql_connect(db_host,db_user,db_passwd, database, sql)
        for i in date:
            del_knowledge(str(i[0]))
