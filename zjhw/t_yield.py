
def test():

    print("第一次打印")
    for i in range(8):
        print(f'当前i的值是： {i}')
        if i == 5:
            yield

    print('循环暂停')


test()


