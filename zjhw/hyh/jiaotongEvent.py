#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import socket

sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
address = ("119.3.193.157", 28888)
sk.connect(address)

body = {
    'deviceId': '62d7c97f6b9813541d525070_2023011503',
    'msgType': 2,
    'msg': {
        'Event': 2,
        'Traffic_light': 1,
        'Device': '62d7c97f6b9813541d525070_2023011503',
        'Plate_number': '浙A·901'
    }
}


#body1 = json.loads(body)
body1 = json.dumps(body)

sk.send(bytes(body1,'utf-8'))
print ("++++++++++11111+++++++++++++")
data = sk.recv(2048)

print(str(data,'utf-8'))

sk.close()
