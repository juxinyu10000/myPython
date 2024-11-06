"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增教师接口
"""
import requests
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from config import username
from config import passwd
from config import host, port
from config import db_host,db_passwd

database="mp-iot"


login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

add_body = {
        "roleId": "955466749301157888",
        "username": "teacher1",
        "realName": "teacher1",
        "phone": "",
        "email": "",
        "pwd": "a1234567.",
        "childIds": ["955498578108411904"],
        "type": "706749786450997348",
        "typeName": "班级"
    }

def add_jiaoShi():
    API = '/service-user/api/v1/user/admin/create'
    url = "https://%s:%d%s" % (host, port, API)
    resp = requests.post(url, json=add_body, headers=header)
    if "4000200" in resp.text:
        #print(resp.json()['data']['username']+"添加成功")
        print (resp.json())
        print ("添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def del_jiaoShi(id):
    API = '/service-user/api/v1/user/admin/delete?id='+id
    url = "https://%s:%d%s" % (host, port, API)

    resp = requests.delete(url,headers=header)
    if "4000200" in resp.text:
        print (resp.json())
        print ("删除成功")
    else:
        print(f"出现了异常，{resp.json()}")
        print (resp.url)

if __name__=='__main__':
    tag=2   # 1 添加教师， 2  删除教师
    zuhu="juxinyu3"
    if tag==1:
        sql = "select * FROM `group` where type_name = '班级' and status=1 and create_user ='"+zuhu+"'"
        date = mysql_connect(db_host, db_passwd, database, sql)
        num=1
        for id in date:
            for i in range (20):
                add_body["realName"] = zuhu+"Techer" + str(num)
                add_body["username"] = zuhu+"_techer" + str(num)
                add_body["childIds"] = [str(id[0])]
                num=num+1
                add_jiaoShi()
    if tag == 2:
        sql="select id from user where username like 'juxinyu3_techer%'"
        date = mysql_connect(db_host, db_passwd, database, sql)
        for i in date:
            del_jiaoShi(str(i[0]))