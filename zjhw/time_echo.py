import time

while True:
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    time.sleep(1)