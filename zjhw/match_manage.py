"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 实验云赛事创建，传参赛事名称
"""

import requests
from zjhw.lab_login import login
from zjhw.DB_connect import mysql_connect
from config import host, port,db_host,db_passwd


username = "juxinyu_admin"
passwd = "a1234567."

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}


def add_match(match_name):
    API = '/service-match/admin/match/info/create'
    url = "https://%s:%d%s" % (host, port, API)
    body = {
        "template": "style1",
        "title": match_name,
        "subtitle": "",
        "logo": "service-aggregation/a3922a4e_11.jpg",
        "startTime": "2022-05-13 14:35:47",
        "endTime": "2023-06-01 14:35:47",
        "place": "",
        "summary": "",
        "tags": [],
        "switches": {
            "bannerSwitch": False,
            "introduceSwitch": False,
            "signUpSwitch": False,
            "questionSwitch": False,
            "arrangeSwitch": False,
            "awardSwitch": False,
            "judgeSwitch": False,
            "unitSwitch": False
        },
        "banners": [],
        "introduce": "",
        "signUpProcess": "",
        "questionDesc": "",
        "arranges": [{
            "name": "开始",
            "startTime": "",
            "endTime": "",
            "explain": "比赛开始"
        }, {
            "name": "结束",
            "startTime": "",
            "endTime": "",
            "explain": "比赛结束"
        }],
        "matchAwardList": [{
            "name": "",
            "desc": ""
        }],
        "judges": [{
            "id": "",
            "introduce": ""
        }],
        "units": [{
            "picture": "",
            "type": 1,
            "name": ""
        }]
    }

    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print("赛事创建成功")
        print(resp.json())
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


def delete_match(match_id):
    API = '/service-match/admin/match/info/delete?id=%s' % match_id
    url = "https://%s:%d%s" % (host, port, API)

    resp = requests.delete(url, headers=header)
    if "4000200" in resp.text:
        print("%s删除成功" % match_id)
        print(resp.json())
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 2
    if tag == 1:
        for i in range(3):
            match_name = "鞠新宇的性能赛事%s" % (i + 1)
            add_match(match_name)
    if tag == 2:
        sql = "select id from match_info where title like '鞠新宇的性能%' AND del_status = 0"
        date = mysql_connect(db_host, db_passwd, "mp_match", sql)
        for match_id in date:
            delete_match(match_id)
