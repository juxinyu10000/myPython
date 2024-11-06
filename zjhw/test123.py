
import pymysql
def update_or_delete_DB(host, passwd, database, sql):


    db = pymysql.connect(host=host, port=3306, user="root", passwd=passwd, database=database)
    # ʹ��cursor()������ȡ�����α�
    cursor = db.cursor()
    try:
        # execute()����ִ��sql
        cursor.execute(sql)
        db.commit()

    except:

        db.rollback()
    finally:
        # �ر��α�����ݿ�����
        cursor.close()
        db.close()


del_sql = "update user set email = '' where email = '916958724@qq.com'"
db_host = '123.60.33.112'
db_passwd = 'Zjhw@123'
update_or_delete_DB(db_host,db_passwd,'mp_user',del_sql)