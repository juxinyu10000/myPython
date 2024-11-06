import requests
from zjhw.talent.talent_login import talen_teacher_login_header
from zjhw.talent.config import host, admin, passwd

body = {
    "account": "huawei1",
    "userName": "",
    "workNumber": "",
    "phone": "",
    "email": "",
    "userStatus": 1,
    "organizationIds": ["847db1a053115f8602f72733af0a74e0"]
}

header = talen_teacher_login_header(host=host, account="juxinyu", passwd="Qwer@123")


def createxy():
    API = "/userauth/user/insertTraineeTypeUser"
    url = "%s%s" % (host, API)
    print (url)
    resp = requests.post(url, json=body, headers=header,verify=False)
    if "SUCCESS" in resp.text:
        print(body['account'] + "新增成功")
    else:
        print(f"出现了异常{resp.json()}")


if __name__ == '__main__':
    shortName = "huawei"
    for i in range(10):
        body['account'] = shortName + str(i + 2)
        createxy()
