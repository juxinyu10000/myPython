"""
encoding = utf-8
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 实验云登录接口，传参用户名和密码
"""

import requests
import json
import redis
from zjhw.config import host
from zjhw.config import redis_host, redis_port, redis_password

header = {
    "Content-Type": "application/json; charset=UTF-8",
}


def get_VerificationCodeId():
    """
    获取滑块验证码
    :return: verificationCodeId
    """
    API = "/service-aggregation/api/v1/verification/code/query"
    url = "%s%s" % (host, API)
    resp = requests.get(url)
    resp_json = json.loads(resp.text)
    return resp_json['data']['verificationCodeId']


def redis_connect(redis_host, redis_port, redis_password):
    """
    连接redis,获取滑块X值，Y值，verificationCodeId
    :param redis_host:
    :param redis_port:
    :param redis_password:
    :return: 字典
    """
    r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, db=8, decode_responses=True)  # 预发环境
    key = f"aggregation:verification:code:{get_VerificationCodeId()}"
    res_redis = eval(r.get(key))
    r.close()
    return res_redis


def login(user, passwd):
    """
    登录接口，返回字典，包括access_token，application_key，groupid，classId
    :param user:  登录用户名
    :param passwd: 登录密码
    :return: 字典
    """
    info = redis_connect(redis_host, redis_port, redis_password)
    API = '/service-user/api/v1/user/pwd/login'
    login_url = "%s%s" % (host, API)
    login_data = {"name": user,
                  "pwd": passwd,
                  "randomId": "",
                  "verificationCodeId": info['verificationCodeId'],
                  "x": info['x'],
                  "y": info['y']
                  }
    resp = requests.post(url=login_url, json=login_data)
    print (resp.json())
    login_info = {}
    if resp.status_code == 200:
        user_token = resp.json()['data']['token']['accessToken']
        applicationKey = resp.json()['data']['groups'][0]['applicationKey']
        groupId = resp.json()['data']['groups'][0]['id']
        #classId = resp.json()['data']['groups'][1]['id']
        # userId=resp.json()['data']['id']
        login_info['access_token'] = user_token
        login_info['application_key'] = applicationKey
        login_info['groupid'] = groupId
        #login_info['classId'] = classId
        # login_info['userId']=userId
        # print(user + "登录成功")
        return login_info
    else:
        print(f"出现了异常，{resp.json()}")
        return


if __name__ == '__main__':
    print(login('juxinyu_admin', 'a1234567.'))
