import requests

header = {"Content-Type": "application/json",
          "access_token": 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJtc2ciOiIiLCJpc1JlZnJlc2giOmZhbHNlLCJleHAiOjE4MDYxMjc2OTMsInVzZXJuYW1lIjoic2R5Z19hZG1pbiJ9.phAJTO4bpayL9oe1qvNWsRUhQBfGhvVLnZw8UpZUrGE',
          "application_key": '1331ec8b-3d40-4d9e-9c33-77a8760fe617',
          "group_id": '1141738674808369152'}

API = '/service-user/api/v1/user/student/create'
host = 'http://syy.sdygxx.com:8097'
body = {"username":"huawei","realName":"","phone":"","pwd":"a1234567.","email":"","code":"","childIds":[],"type":"706749786450997348","typeName":"班级"}
url = "%s%s" % (host, API)
for i in range(199):
    body["username"] = "huawei" + str(i+2)
    resp = requests.post(url, json=body, headers=header)
    if resp.status_code == 200:
        print( body["username"])
    else:
        print(resp.text)