"""
encoding = utf-8
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 深度学习登录
"""

import requests
import hashlib
from zjhw.config import host
from zjhw.config import username, passwd

header = {
    "Content-Type": "application/json;charset=UTF-8",
}


def login(user, pwd):
    API = "/service-user/open/user/login"
    login_url = "%s%s" % (host, API)
    m = hashlib.md5()  # 是用来创建MD5加密对象
    m.update(pwd.encode("utf8"))  # 是把pwd字符用MD5算法加密
    login_data = {
        "key": user,
        "pwd": m.hexdigest()
    }

    resp = requests.post(url=login_url, json=login_data, headers=header)
    if resp.status_code == 200:
        print(user + "登录成功")
        lista = []
        token = resp.json()['data']['token']['accessToken']
        userid = resp.json()['data']['id']
        lista = [token,userid]
        return lista
    else:
        print(f"出现了异常，{resp.json()}")
        return


if __name__ == '__main__':
    print(login("xueyuan5", "a123456."))
