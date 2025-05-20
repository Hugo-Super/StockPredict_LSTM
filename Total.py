import datetime
import akshare as ak
import sqlite3
import re
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split



def get_history_datas(code,start_date):
    '''
    获取股票历史数据功能
    :param code: 股票代码格式 sz000001
    :param start_date:开始时间格式 19910403
    :return:dataframe数据对象  date open high low close volume amount outstanding_share turnover
    '''
    start_date = start_date # 设定获取日线行情的初始日期和终止日期，其中终止日期设定为昨天。
    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = time_temp.strftime('%Y%m%d')

    df = ak.stock_zh_a_daily(symbol=code, start_date=start_date, end_date=end_date)
    return df


def init_stockdb(code,dbpath):
    '''
    创建数据库功能（包含建表）
    :param code: 股票代码，会以此名建表
    :param dbpath: 数据库路径
    :return:
    '''
    sql = '''
            create table if not exists {}
            (
            state_date varchar(45) primary key,
            open decimal(20,2),
            high decimal(20,2),
            low decimal(20,2),
            close decimal(20,2),
            volume int(20),
            amount decimal(30,2),
            outstanding_share decimal(30,2),
            turnover decimal(15,10),
            stock_code varchar(45)                      
            );
        '''
    sql = sql.format(code)# 创建数据表
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    conn.close()


def save_history_db(code,df,dbpath):
    '''
    导入数据库功能
    :param code: 股票代码
    :param df: dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param dbpath: 数据库路径
    :return:
    '''
    df['stock_code'] = code
    c_len = df.shape[0]
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()
    for j in range(c_len):
        m = c_len-1-j
        resu0 = list(df.iloc[c_len-1-j])#.ix
        resu = []
        for k in range(len(resu0)):
            if str(resu0[k]) == 'nan':
                resu.append(-1)
            else:
                resu.append(resu0[k])
        state_dt = resu[0].strftime('%Y-%m-%d') # resu[0]返回的是datetime格式
        try:
            sql_insert = "INSERT INTO {}(state_date,open,high,low,close,volume,amount,outstanding_share,turnover,stock_code) VALUES ('%s','%.2f','%.2f','%.2f','%.2f','%i','%.2f','%.2f','%.2f','%s')" % (state_dt,float(resu[1]),float(resu[2]),float(resu[3]),float(resu[4]),float(resu[5]),float(resu[6]),float(resu[7]),float(resu[8]),str(resu[9]))
            sql_insert = sql_insert.format(code)
            cur.execute(sql_insert)
            conn.commit()
        except Exception as err:
            continue
    cur.close()
    conn.close()
    print("all finish!")




def Stock_Price_LSTM_Data_Precesing(df, mem_his_days, pre_days):
    '''
    数据预处理功能，对获取的数据进行预处理
    :param df: dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param mem_his_days: 记忆天数
    :param pre_days: 预测天数
    :return: 清洗好的数据X, y, X_lately
    '''
    # 删除空值
    df.dropna(inplace=True)
    # 按交易日期升序
    df = df.sort_values(by='date')

    # 建标签，以收盘价为准 预测10天 故往上推10天
    df['label'] = df['close'].shift(periods=-pre_days)
    # print(df)

    # 对数据标准化
    scaler = StandardScaler()
    # X取1到6列为特征向量，即open,high,low,close,volume
    sca_X = scaler.fit_transform(df.iloc[:, 1:6])
    # print(sca_X)

    deq = deque(maxlen=mem_his_days)

    X = []
    for i in sca_X:
        deq.append(list(i))
        if len(deq) == mem_his_days:
            X.append(list(deq))

    X_lately = X[-pre_days:]
    X = X[:-pre_days]
    # print(len(X))
    # print(len(X_lately))

    y = df['label'].values[mem_his_days - 1:-pre_days]
    # print(len(y))

    # 制成numpy格式
    X = np.array(X)
    y = np.array(y)
    # print(X.shape)
    # print(y.shape)

    return X, y, X_lately



def train_LSTM_model(df,mem_days,pre_days,lstm_layers,dense_layers,units):
    '''
    训练模型并生成的功能
    :param df:dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param mem_days: 记忆天数
    :param pre_days: 预测天数
    :param lstm_layers: 长短期记忆模型层数
    :param dense_layers: 全连接层数
    :param units:神经元数
    :return:
    '''
    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            for the_dense_layers in dense_layers:
                for the_units in units:
                    # 模型存储路径 评价(平均误差率) 期数 记忆天数 lstm层数 全连接层数 神经元数(.keras   .weights.h5)
                    filepath = './models/{val_mape:.2f}_{epoch:02d}_' + f'mem_{the_mem_days}_pre_{pre_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}' + '.h5'
                    checkpoint = ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode='min',
                        save_best_only=True)
                    # print(filepath)

                    X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)

                    # 划分训练集和测试集
                    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

                    # 构建LSTM的神经网络
                    model = Sequential()
                    # 输入层
                    model.add(keras.Input(shape=X.shape[1:]))
                    model.add(LSTM(the_units, activation='relu', return_sequences=True))
                    model.add(Dropout(0.1))  # 防过拟合

                    for i in range(the_lstm_layers):
                        model.add(LSTM(the_units, activation='relu', return_sequences=True))
                        model.add(Dropout(0.1))  # 防过拟合

                    model.add(LSTM(the_units, activation='relu'))
                    model.add(Dropout(0.1))  # 防过拟合

                    # 全连接层
                    for i in range(the_dense_layers):
                        model.add(Dense(the_units, activation='relu'))
                        model.add(Dropout(0.1))  # 防过拟合

                    # 输出层
                    model.add(Dense(1))
                    # 优化器：adam，损失函数：MeanSquaredError平均方差，评价函数:MeanAbsolutePercentageError
                    # Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳.
                    # 回归问题
                    model.compile(optimizer='adam', loss='mse', metrics=['mape'])

                    # 训练 批大小32 训练期数50 测试样本test
                    model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                              callbacks=[checkpoint])

# TODO 路径选择
def get_path():
    """注意，以下列出的方法都是返回字符串而不是数据流"""
    # 返回一个字符串，且只能获取文件夹路径，不能获取文件的路径。
    # path = filedialog.askdirectory(title='请选择一个目录')

    # 返回一个字符串，可以获取到任意文件的路径。
    path = filedialog.askopenfilename(title='请选择文件')

    # 生成保存文件的对话框， 选择的是一个文件而不是一个文件夹，返回一个字符串。
    # path = filedialog.asksaveasfilename(title='请输入保存的路径')

    #entry_text.set(path)

# 导入模型进行预测的功能

def model_loading(modelpath):
    best_model = load_model(modelpath)

    mday = re.findall('(?<=mem_)\d+', modelpath) #从文件名匹配记忆天数
    the_mem_days = mday[0]
    pday = re.findall('(?<=pre_)\d+', modelpath)  # 从文件名匹配预测天数
    pre_days = pday[0]
    X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    best_model.evaluate(X_test, y_test)
    pre = best_model.predict(X_test)

    df_time = df.iloc[-len(y_test):, 0]
    plt.figure(dpi=300, figsize=(24, 8))
    plt.plot(df_time, y_test, color='red', label='price')
    plt.plot(df_time, pre, color='green', label='predict')
    plt.xticks(rotation=90)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    # plt.gcf().autofmt_xdate()
    plt.show()


def stock_trend(df,funtioncode1):
    '''
    股票趋势图功能
    :param df: dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param funtioncode: 功能码1看历史全部，2近十年，3近十二个月，4近一个季度，5近一个月，6近七天
    :return:
    '''
    plt.figure(dpi=300, figsize=(24, 8))
    if funtioncode1 == 1:#看历史全部
        df_y_close = df['close'].values
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.show()
    elif funtioncode1 == 2:#近十年
        df_y_close = df['close'].values[-3650:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.show()
    elif funtioncode1 == 3:#近十二个月
        df_y_close = df['close'].values[-365:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.show()
    elif funtioncode1 == 4:#近一个季度
        df_y_close = df['close'].values[-90:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.show()
    elif funtioncode1 == 5:#近30天
        df_y_close = df['close'].values[-30:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.show()
    elif funtioncode1 == 6:#近七天
        df_y_close = df['close'].values[-7:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.show()


def kline(df,funtioncode2):
    '''
    股票K线图功能（均线）
    :param df: dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param funtioncode2: 功能码 1近30天K线，2近90天K线，3近1年K线
    :return:
    '''
    df_cleaning = df.iloc[:, :6]
    df_cleaning['date'] = pd.to_datetime(df_cleaning['date'])
    df_cleaned = df_cleaning.set_index(keys='date')

    my_color = mpf.make_marketcolors(
        up="red",  # 上涨K线的颜色
        down="green",  # 下跌K线的颜色
        edge="black",  # 蜡烛图箱体的颜色
        wick="black",  # 蜡烛图影线的颜色
        volume="inherit"  # 继承up和down的颜色
    )

    # 设置图表的背景色
    my_style = mpf.make_mpf_style(
        base_mpl_style='seaborn',
        marketcolors=my_color,
        rc={'font.family': 'SimHei', 'axes.unicode_minus': 'False'}
    )
    if funtioncode2 == 1:
        mpf.plot(df_cleaned.iloc[-30:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近30天K线图（均线5日、10日）',
                 mav=(5, 10),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")
    elif funtioncode2 == 2:
        mpf.plot(df_cleaned.iloc[-90:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近90天K线图（均线5日、10日）',
                 mav=(5, 10),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")
    elif funtioncode2 == 3:
        mpf.plot(df_cleaned.iloc[-365:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近1年K线图（均线20日、60日）',
                 mav=(20, 60),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")


# TODO 股票龙虎榜功能（查接口）
def dataframe_to_treeview(dfs, x1, y1, w, h, column_name='序号'):
    # 1.获取数据的列标题
    a = dfs.columns.values.tolist()
    a.insert(0, column_name)
    # 添加一个宽度列表，组成字典
    b = [80 for nums in range(len(a) - 1)]
    # [50, 80, 80, 80, 80, 80]
    b.insert(0, 50)
    df_titles = dict(zip(a, b))
    # print(df_titles)
    # 2.设置纵向滚动条
    xbar = tk.Scrollbar(frame1, orient='horizontal')
    xbar.place(x=x1, y=y1 + h - 3, width=w)
    ybar = tk.Scrollbar(frame1, orient='vertical')
    ybar.place(x=x1 + w - 3, y=y1, height=h)
    # 3.创建Treeview
    tree = ttk.Treeview(frame1, show='headings',
                        xscrollcommand=xbar.set,
                        yscrollcommand=ybar.set)

    tree['columns'] = list(df_titles)
    # 批量设置列属性
    for title in df_titles:
        # 加载列标题
        tree.heading(title, text=title)
        tree.column(title, width=df_titles[title], anchor='center')

    # 遍历DataFrame的每一行，并将它们添加到Treeview中
    for index, row in dfs.iterrows():
        datas = row.tolist()
        datas.insert(0, index)
        # print(datas)
        # 添加行数据
        tree.insert('', 'end', text='', values=datas)
    # 将Treeview添加到主窗口
    tree.place(x=x1, y=y1, width=w, height=h)
    xbar.config(command=tree.xview)
    ybar.config(command=tree.yview)


def stock_lhb():
    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = time_temp.strftime('%Y%m%d')
    stock_lhb_detail_daily_sina_df = ak.stock_lhb_detail_daily_sina(date=end_date)
    df = stock_lhb_detail_daily_sina_df.iloc[:, 1:]
    df_rise = df[df['指标'] == '涨幅偏离值达7%的证券'].iloc[:, :6]
    df_fall = df[df['指标'] == '跌幅偏离值达7%的证券'].iloc[:, :6]
    dataframe_to_treeview(df_rise, 180, 50, 600, 200)
    dataframe_to_treeview(df_fall, 180, 300, 600, 200)
# TODO 股票实时查看功能

if __name__ == '__main__':

    dbpath = 'stock.db'
    code = 'sz000001'
    start_date = '19910403'
    init_stockdb(code,dbpath)
    df = get_history_datas(code,start_date)
    #save_history_db(code, df, dbpath)
    pre_days = 10
    mem_days = [15]
    lstm_layers = [1]
    dense_layers = [1]
    units = [32]
    train_LSTM_model(df,mem_days,pre_days,lstm_layers,dense_layers,units)




