"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 新增学员接口
"""
import requests
from zjhw.DB_connect import mysql_connect
from zjhw.lab_login import login
from zjhw.config import host
from zjhw.config import db_host,db_user,db_passwd,database

host = host
username = "juxinyu1"
passwd = "a1234567."

db_host = db_host
db_passwd = db_passwd
database = 'mp_user'
db_user = db_user

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

add_body = {
    "username": "juxinyu_student1",
    "realName": "juxinyuStudent1",
    "phone": "",
    "pwd": "a1234567.",
    "email": "",
    "code": "",
    "childIds": ["955520897233121280"],
    "type": "706749786450997348",
    "typeName": "班级"
}

del_body=[]

def add_xueYuan():
    API = '/service-user/api/v1/user/student/create'
    url = "https://%s:%d%s" % (host, port, API)
    resp = requests.post(url, json=add_body, headers=header)
    if "4000200" in resp.text:
        print(resp.json()['data']['username']+"添加成功")
        print("添加成功")
    else:
        print(f"出现了异常，{resp.json()}")


def del_xueYuan():
    API = '/service-user/api/v1/user/student/delete'
    url = "%s%s" % (host, API)
    print (url)
    resp = requests.delete(url,json=del_body, headers=header)
    if "4000200" in resp.text:
        print(resp.json())
        print("删除成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__=='__main__':
    tag=2   # 1 添加学员， 2  删除学员
    zuhu="juxinyu1"
    if tag==1:
        #sql = "select id FROM `group` where type_name = '班级' and status=1 and name LIKE '测试%'"  # 10个班级
        #sql = "select id FROM `group` where type_name = '班级' and status=1 and name = '测试6班'"   # 1个班
        sql = "select id FROM `group` where type_name = '班级' and status=1 and create_user ='"+zuhu+"'"
        date = mysql_connect(db_host,db_user,db_passwd, database, sql)
        num=1
        for id in date:
            for i in range (800):  #每个班级800个学员
                add_body["realName"] = zuhu+"Student" + str(num)
                add_body["username"] = zuhu+"_student" + str(num)
                add_body["childIds"] = [str(id[0])]
                num=num+1
                add_xueYuan()
    if tag == 2:
        sql="select id from user where username like 'juxinyu_student%'"
        date = mysql_connect(db_host,db_user,db_passwd, database, sql)
        for i in date:
            del_body=[i[0]]
            del_xueYuan()
