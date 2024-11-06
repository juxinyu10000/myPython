"""
绑定学员和考试计划
"""
import requests
from zjhw.lab_login import login
from zjhw.DB_connect import mysql_connect
from config import host, db_host, db_passwd,db_user,username,passwd

username = username
passwd = passwd

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

body = [{
    "evaluationPlanId": "976509808906141696",
    "userId": "955540607001620480",
    "implementId": "976509808906141697"
}]


def bind_student():
    API = '/service-exam/student/import'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print("绑定成功")
        return resp.json()['data']
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        sql = "select id from user where username like '%juxinyu_student%'"
        date = mysql_connect(db_host,db_user,db_passwd, "mp_user", sql)
        num = 1
        for i in date:
            body[0]["userId"] = i[0]
            print(num)
            bind_student()
            num = num + 1
