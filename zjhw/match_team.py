"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 实验云赛事队伍创建
"""

import requests
from zjhw.lab_login import login
from zjhw.DB_connect import mysql_connect
from config import host, db_host, db_passwd,db_user

username = "admin"
passwd = "a1234567."

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

body = {
    "teamName": "1队",
    "teamNo": "101",
    "unit": "",
    "userIdList": ["1011657621738262528", "1011657621658570752", "1011657621813760000"],
    "captain": "",
    "matchId": "1016730106183487488"
}


def add_team():
    API = '/service-match/admin/match/team/create'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    print (body)
    if "4000200" in resp.text:
        print("参赛队伍创建成功")
        print(resp.json())
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


del_body = {
    "teamIdList": ["974700126041862144"]
}


def del_team():
    API = '/service-match/admin/match/team/delete'
    url = "%s%s" % (host, API)
    resp = requests.delete(url, json=del_body, headers=header)
    if "4000200" in resp.text:
        print("队伍删除成功")
        print(resp.json())
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    match_id = "1015300835535618048"
    tag = 1
    if tag == 1:
        sql = "select user_id from mp_match.match_user where  user_name like '%juxinyu1_student%' and match_id = " \
              "'1015300835535618048' "
        date = mysql_connect(db_host,db_user, db_passwd, "mp_match", sql)

        num = 1
        for i in date:
            body["teamName"] = "性能测试赛事队伍%d" % num
            body["userIdList"] = [i[0]]
            body["matchId"] = match_id
            add_team()
            num = num + 1

    elif tag == 2:
        sql = "select id from match_team where match_id='" + match_id + "' and del_status = 0"
        date = mysql_connect(db_host, db_user,db_passwd, "mp_match", sql)
        for i in date:
            del_body["teamIdList"] = [i[0]]
            del_team()
