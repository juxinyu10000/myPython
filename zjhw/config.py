tag = 5

if tag == 4:
    host = "http://gateway.test.labcloud.com"
    # redis数据库
    redis_host = "192.168.77.64"
    redis_password = "Redis@91"
    redis_port = 6379
    # mysql数据库
    db_host = '192.168.77.69'
    db_user = "nacos"
    db_passwd = 'Zjhw@123'
    database = 'mp_user'

    username = "zd050601"
    passwd = "a1234567."
elif tag == 5:
    # 公有云稳定性环境
    # 域名&端口
    host = "https://tc-stable-portal.zj-huawei.com:28443"
    # redis数据库
    redis_host = "124.71.151.53"
    redis_password = "Redis@91"
    redis_port = 8080
    # mysql数据库
    db_host = '123.60.33.112'
    db_user = "root"
    db_passwd = 'Zjhw@123'
    database = 'mp_user'

    username = "juxinyu_admin"
    passwd = "a1234567."

elif tag == 6:
    # 深度学习GPU
    # 域名&端口
    host = "http://192.168.123.165:31000"

    # mysql数据库
    db_host = '192.168.123.171'
    db_user = "root"
    db_passwd = 'ZJHW_dl-dlvr@800'
    database = 'arch-ai-deeplearn'

    username = "juxinyu"
    passwd = "a123456."


elif tag == 7:
    # 深度学习NPU
    # 域名&端口
    host = "http://192.168.88.74:31000"

    # mysql数据库
    db_host = '192.168.88.70'
    db_user = "root"
    db_passwd = 'ZJHW_dl-dlvr@800'
    database = 'arch-user'

    username = "juxinyu"
    passwd = "a1234567."
