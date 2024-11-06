#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 批量修改图片名称
"""

import os
import random
import shutil
import string


path = "D:\pic"
file_list = os.listdir(path)
filename1 = path + '\\' + file_list[0]
i=0
while i < 1001:
    value = ''.join(random.sample(string.ascii_letters, 5))
    filename2 = path + '\\' + value + file_list[0]
    print (filename2)
    shutil.copyfile(filename1,filename2)
    i = i + 1