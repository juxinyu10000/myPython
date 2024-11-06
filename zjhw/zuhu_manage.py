import requests
from zjhw.lab_login import login
from config import username
from config import passwd
from config import host, port


login_info = login(username, passwd)
header = {"Content-Type": "application/json",
              "access_token": login_info['access_token'],
              "application_key": login_info['application_key'],
              "group_id": str(login_info['groupid'])}
print (header)

def add_zuHu(num):
    addZuhu_API1 = '/api/v1/group/create'  # 保存组织(可保存租户和班级)
    addZuhu_url1 = "https://%s:%d%s" % (host, port, addZuhu_API1)
    body1 = {
        "id": "",
        "type": "606749786450997348",
        "typeName": "租户",
        "name": "性能测试租户%d" % (num),
        "pic": "",
        "startTime": "2022-03-21 00:00:00",
        "endTime": "2022-04-30 23:59:59",
        "domain": "",
        "tenantCount": "9999"
    }
    resp1 = requests.post(addZuhu_url1, json=body1, headers=header)
    if resp1.status_code == 200:
        print("成功添加租户组织")
        id = resp1.json()['data']['id']
    else:
        print(f"出现了异常，{resp1.json()}")

    addZuhu_API2 = '/service-edumanager/api/v1/edu/manager/create'  # 保存租户的基本信息
    addZuhu_url2 = "https://%s:%d%s" % (host, port, addZuhu_API2)
    body2 = {
        "id": id,
        "typeName": "租户",
        "name": "性能测试租户%d" % (num),
        "logoUrl": "",
        "startTime": "2022-03-21 00:00:00",
        "endTime": "2022-04-30 23:59:59",
        "domain": ""
    }
    resp2 = requests.post(addZuhu_url2, json=body2, headers=header)
    if resp2.status_code == 200:
        print("保存租户的基本信息成功")
    else:
        print(f"出现了异常，{resp2.json()}")

    addZuhu_API3 = '/service-edumanager/api/v1/edu/manager/create'  # 更新租户核时
    addZuhu_url3 = "https://%s:%d%s" % (host, port, addZuhu_API3)
    body3 = {
        "groupId": id,
        "groupName": "性能测试租户%d" % (num),
        "id": "955467420121542656",
        "totalCoreTime": "",
        "type": 2
    }
    resp3 = requests.put(addZuhu_url3, json=body3, headers=header)
    if resp3.status_code == 200:
        print("更新租户核时成功")
    else:
        print(f"出现了异常，{resp3.json()}")


def add_zuhu_gunaLiYuan(num):
    addZuhu_API = '/service-user/api/v1/user/admin/create'  # 创建租户管理员
    addZuhu_url = "https://%s:%d%s" % (host, port, addZuhu_API)
    body = {
        "childIds": ["955467419643211776"],
        "username": "juxinyu%d" % (num),
        "realName": "",
        "phone": "",
        "email": "",
        "pwd": "a1234567.",
        "type": "606749786450997348",
        "typeName": "租户"
    }
    resp = requests.put(addZuhu_url, json=body, headers=header)
    if resp.status_code == 200:
        print(" 创建租户管理员成功")
    else:
        print(f"出现了异常，{resp.json()}")

def query_zuhu_gunaLiYuan(keyword):
    queryZuhu_API = "/service-user/api/v1/user/all/account/query?primaryAccount=0&groupId=955479073634316288&groupName=&username="+keyword+"&realName=&phone=&updateUser=&pageNum=1&pageSize=10&createTimeSort=" # 查询租户管理员
    queryZuhu_url = "https://%s:%d%s" % (host, port, queryZuhu_API)
    resp = requests.get(queryZuhu_url, header)
    if resp.status_code == 200:
        print(" 查询租户管理员成功")
        print (resp.json())
        id = resp.json()['data']['list'][0]['id']
        return id

    else:
        print(f"出现了异常，{resp.json()}")





if __name__=='__main__':
    print (query_zuhu_gunaLiYuan('juxinyu5'))