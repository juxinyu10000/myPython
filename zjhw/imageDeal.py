#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 图片处理
"""
import os
import requests
import json
import logging
import logging.config
import base64


host="https://tc-pre-portal.zj-huawei.com:28443"

root_path = "D:/myPython/zjhw/test_image/"



def get_VerificationCodeId():
    """
    获取滑块验证码
    :return: verificationCodeId
    """
    API = "/service-aggregation/api/v1/verification/code/query"
    url = "%s%s" % (host, API)
    resp = requests.get(url)
    resp_json = json.loads(resp.text)
    change_img(resp_json['data']['shadeImage'],"大图1")
    change_img(resp_json['data']['cutoutImage'], "小图1")
    return resp_json['data']['verificationCodeId']

def change_img(base64_img, name_img):
    """
    把base64的图片改成png图片,并保存在本地
    :param base64_img: base64格式的图片
    :param name_img: 保存的图片名称
    :return:
    """
    logging.info(name_img)

    image = base64.b64decode(base64_img)
    path_yaml = root_path
    if os.path.exists(path_yaml):
        logging.info(r"保存图片/test_img/{}.png成功".format(name_img))
        with open(path_yaml + r"{}.png".format(name_img), 'wb') as f:
            f.write(image)
    else:
        logging.error("登录时候保存图片失败，请检查路径")

get_VerificationCodeId()