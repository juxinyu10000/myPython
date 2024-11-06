import random
import logging
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import time

logging.basicConfig(level=logging.DEBUG)


def nowTime(request):
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return HttpResponse(now_time)


def getUser(request):
    name = [["Lucas", 5, 'boy'], ["Anan", 5, 'boy'], ["Peter", 6, 'boy'], ['Jack', 6, 'boy'], ['Amy', 6, 'girl'],
            ['Emma', 7, 'girl'], ['grace', 6, 'girl']]
    stu = random.choice(name)
    logging.info(stu)

    date = {
        "code": 200,
        "message": "success",
        "data": {
            "Class": 'A1038',
            "name": stu[0],
            'age': stu[1],
            'sex': stu[2],
            'test': {
                'edi': 'Python',
                'game': 'game',
                'book': 'book',
                'travel': 'travel'
            }
        }
    }

    return JsonResponse(date)


def helloname(request):
    context = {}
    context['helloname'] = 'lucas'
    return render(request, 'helloname.html', context)
