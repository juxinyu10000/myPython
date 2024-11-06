import requests
from zjhw.lab_login import login
from zjhw.config import username
from zjhw.config import passwd
from zjhw.config import host


def dev_bind(nodeId):
    API = "/service-iot-lab/api/v1/device/bind?nodeId=" + nodeId
    url = "%s%s" % (host, API)
    resp = requests.post(url, headers=header)
    if "4000200" in resp.text:
        print(username + "和设备" + nodeId + "绑定成功")
    else:
        print(f"出现了异常，{resp.json()}")


if __name__ == '__main__':
    for i in range(150):
        username = "juxinyuxy" + str(i + 1)
        str_num = str(i + 1).zfill(3)
        login_info = login(username, passwd)
        header = {"Content-Type": "application/json",
                  "access_token": login_info['access_token'],
                  "application_key": login_info['application_key'],
                  "group_id": str(login_info['groupid'])}
        if i < 50:
            nodeId = "202304131600" + str_num
        elif 50 <= i <= 150:
            nodeId = "202304130160" + str_num

        dev_bind(nodeId)