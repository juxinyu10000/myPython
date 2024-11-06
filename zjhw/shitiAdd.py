"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 试题管理
"""

import requests
from zjhw.lab_login import login

from config import host

username = "juxinyu1"
passwd = "a1234567."

login_info = login(username, passwd)
header = {"Content-Type": "application/json",
          "access_token": login_info['access_token'],
          "application_key": login_info['application_key'],
          "group_id": str(login_info['groupid'])}

body = {
    "catalogId": "1016349873139204096",
    "knowledgeId": "",
    "tags": [],
    "type": "1",
    "difficultyLevel": "1",
    "title": "<p>单选题-性能测试如何做</p>",
    "answerParse": "",
    "optionItems": ["A、<p>专业人去做</p>", "B、<p>你说的算</p>", "C、<p>哈哈</p>",
                    "D、<pre class=\"ql-syntax\" spellcheck=\"false\">l拉拉\n</pre>"],
    "answer": ["A"],
    "id": "0"
}


def add_shiti():
    API = '/service-question/api/v1/questions/create'
    url = "%s%s" % (host, API)
    resp = requests.post(url, json=body, headers=header)
    print(body)
    if "4000200" in resp.text:
        print("试题添加成功")
        print(resp.json())
        return resp.json()['data']
    else:
        print(f"出现了错误，{resp.json()}")


if __name__ == '__main__':
    tag = 1
    if tag == 1:
        for i in range(100):
            body["title"] = "<p>单选题-性能测试单选题-" + str(i + 1) + "</p>"
            add_shiti()
