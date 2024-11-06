"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 添加班级接口，传参班级名称
"""
import requests
from zjhw.lab_login import login
from config import username
from config import passwd
from config import host

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}


def add_banJi(banji):
    API = '/service-user/api/v1/group/create'
    url = "https://%s:%d%s" % (host, API)
    body = {
        "type": "706749786450997348",
        "typeName": "班级",
        "name": banji
    }
    resp = requests.post(url, json=body, headers=header)
    if "4000200" in resp.text:
        print("班级添加成功")
        print(resp.json())
        return resp.json()['data']['id']
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    zuhu = "juxinyu4"
    for i in range(10):
        banji = zuhu + "测试%d班" % (i + 1)
        add_banJi(banji)
