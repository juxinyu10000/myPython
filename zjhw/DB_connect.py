"""
@Author  : 鞠新宇
@Contact : JWX1144744
@Function: 连接mysql
"""
import pymysql
from zjhw.config import db_host, db_user,db_passwd, database


def mysql_connect(host,passwd, database, sql):
    # 打开数据库连接
    db = pymysql.connect(host=host, port=3306, user="root", passwd=passwd, database=database)
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    # execute()方法执行sql
    cursor.execute(sql)
    # fetchall()方法获取查询结果
    date = cursor.fetchall()
    # 关闭游标和数据库连接
    cursor.close()
    db.close()
    return date


if __name__ == '__main__':
    # sql = "select id,NAME FROM `group` where type_name = '班级' and status=1 and name LIKE '测试%'"
    sql = "update user set email = '' where email = 'juxinyu10000@163.com'"
    date = mysql_connect(db_host,db_passwd, database, sql)
    print(date)
