import os.path
import sqlite3

def main():

    dbpath = r"D:\PycharmProjects\pythonProject\stock.db"
    init_db(dbpath)


def init_db(dbpath):
    sql = '''
        create table if not exists stock_all
        (
        state_dt varchar(45) primary key,
        stock_code varchar(45),
        open decimal(20,2),
        close decimal(20,2),
        high decimal(20,2),
        low decimal(20,2),
        vol int(20),
        amount decimal(30,2),
        pre_close decimal(20,2),
        amt_change decimal(20,2),
        pct_change decimal(20,2),
        index(stock_code)
        );
    '''        #创建数据表
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    conn.close()
    print("Finish")

if __name__ == "__main__":
    main()

sql = '''
        create table if not exists model_ev_mid
        (
        state_dt varchar(45) primary key,
        stock_code varchar(45),
        resu_predict decimal(20,2),
        resu_real decimal(20,2),
        index(stock_code)
        );
    '''