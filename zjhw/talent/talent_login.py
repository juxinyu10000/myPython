#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/1/18
@Author  : 杨良佳
@Contact : ywx1095537
@File    : talent_login.py
@Function: 
"""
import requests
import ddddocr
import logging
from zjhw.talent.config import host

login_head = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/92.0.4515.159 Safari/537.36"
}

logging.getLogger().setLevel(logging.INFO)  # 定义日志级别
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s")


def login_img_code():
    """
    登录第一步，获取图形验证码
    :return:
    """
    url = host + "/userauth/auth/getImageVerificationCode"
    res = requests.get(url, verify=False, timeout=20)  # 登录的图片验证码
    session = res.headers['Set-Cookie'].split('SESSION=')[1].split(';')[0]
    if res.status_code == 200:
        img_bytes = res.content  # 返回二进制的内容
        ocr = ddddocr.DdddOcr()
        img_code = ocr.classification(img_bytes)
        # print("获取登录的图片验证码结果为：", img_code)
        return img_code, session
    else:
        logging.error("获取登录的图片验证码接口异常")
        # 获取失败后使用万能验证码
        return 1111, session


def get_CsrfToken(headers):
    """
    登录后获取csrfToken
    :param headers:
    :return:
    """
    url = host + "/userauth/admin/auth/getCsrfToken"
    res = requests.get(url, headers=headers, verify=False, timeout=20)  #
    if res.json()['code'] == "EDU00000":
        csrfToken = res.json()["data"].get("csrfToken")  # 返回二进制的内容
        logging.info("getCsrfToken请求成功")
        return csrfToken
    else:
        logging.error("getCsrfToken接口异常")
        return 1


def talen_teacher_login_header(account, passwd):
    """
    根据账号和密码返生成一个固定的登录状态请求头
    :param account:
    :param passwd:
    :return:
    """
    verificationCode, login_session = login_img_code()  # 使用图片识别获取验证码
    # verificationCode = "1111"
    url = host + "/userauth/admin/auth/login"
    headers = {
        'Connection': 'close',
        'Content-Type': 'application/json',
        "Cookie": f'OLE_SESSION = {login_session}'
    }
    data = {
        'account': account,
        'password': passwd,
        'verificationCode': verificationCode}
    res = requests.post(url, json=data, headers=headers, verify=False)
    if res.json()['code'] == "EDU00000":
        login_success_data = res.json()
        logging.info(f"登录成功，登录返回{login_success_data}")
        ole_session = res.headers['Set-Cookie'].split('OLE_SESSION=')[1].split(';')[0]
        refresh_token = res.headers['Set-Cookie'].split('refresh_token=')[1].split(';')[0]
        xcsrf_token = res.headers['Set-Cookie'].split('XSRF-TOKEN=')[1].split(';')[0]
        new_headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Refresh-Token': refresh_token,
            'Cookie': f'XSRF-TOKEN={xcsrf_token}; OLE_SESSION={ole_session}; refresh_token={refresh_token}'
        }
        csrfToken = get_CsrfToken(headers=new_headers)
        login_headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Auth-Token': csrfToken,
            'X-XSRF-TOKEN': xcsrf_token,
            'Cookie': f'XSRF-TOKEN={xcsrf_token}; OLE_SESSION={ole_session}; refresh_token={refresh_token}'
        }
        return login_headers
    else:
        logging.error(f"登录失败，登录结果为：{res.json()}")
        return None


if __name__ == '__main__':
    user_login_headers = talen_teacher_login_header(account="juxinyuAdmin", passwd="Qwer@123")

