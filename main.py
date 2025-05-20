import random
import akshare as ak
import tkinter
import tkinter.messagebox
import tkinter.ttk
from tkinter import filedialog
import datetime

import sqlite3
import re

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

global df
global code
global dbpath


def load_data():
    with open('./res/data.csv', 'r') as infile:  # 打开文件
        return map(str.strip, infile.readlines())  # 返回处理后数据


def toplevel_register() -> None:
    register = tkinter.Toplevel(login)  # 创建注册窗口
    register.title('用户注册')  # 注册窗口标题
    register.geometry('250x125+500+300')  # 注册窗口大小及位置
    register.resizable(False, False)  # 设定注册窗口大小不可改变

    tkinter.Label(register, text='用户名').place(width=50, height=25, x=25, y=5)  # “用户名”文字标签
    tkinter.Label(register, text='新密码').place(width=50, height=25, x=25, y=35)  # “新密码”文字标签
    tkinter.Label(register, text='新密码').place(width=50, height=25, x=25, y=65)  # “重复新密码”文字标签
    (account := tkinter.ttk.Entry(register)).place(width=150, height=25, x=80, y=5)  # 新用户名输入框
    (password := tkinter.ttk.Entry(register, show='●')).place(width=150, height=25, x=80, y=35)  # 新密码输入框
    (password_ := tkinter.ttk.Entry(register, show='●')).place(width=150, height=25, x=80, y=65)  # 重复密码输入框
    tkinter.ttk.Button(register, text='注册', command=lambda: register_account()).place(width=100, height=27, x=20, y=94)  # 注册按钮
    tkinter.ttk.Button(register, text='取消', command=register.destroy).place(width=100, height=27, x=130, y=94)  # 登录按钮

    def register_account() -> None:
        if not (account.get() and password.get()):  # 用户名或密码为空
            tkinter.messagebox.showwarning('注册提示', '用户名或密码不可为空！')
        elif password.get() != password_.get():  # 两次密码不一致
            tkinter.messagebox.showwarning('注册提示', '两次密码不一致！')
        elif account.get() in [line.split(',')[0] for line in load_data()]:  # 用户名已被注册
            tkinter.messagebox.showerror('注册提示', '用户名已被注册！')
        else:  # 注册成功
            with open('res/data.csv', 'a') as infile:  # 打开文件
                infile.write('%s,%s\n' %(account.get(), password.get()))  # 写入信息
            tkinter.messagebox.showinfo('注册提示', '注册成功！')
            register.destroy()  # 关闭注册窗口


def test_for_password(count: list[int] = [0]) -> None:
    if not (account.get() and password.get()):  # 用户名或密码为空
        tkinter.messagebox.showwarning('登录提示', '用户名或密码不可为空！')
    elif account.get()+','+password.get() in load_data():  # 登录成功
        tkinter.messagebox.showinfo('登录提示', '登录成功！')
        login.destroy()  # 摧毁登录窗口
        root.overrideredirect(False)  # 显示主窗口外框
        root.geometry('960x540')  # 重新设置主窗口大小及位置


    else:  # 用户名或密码错误
        count[0] += 1  # 错误计数
        if count[0] < 5:  # 错误适量
            tkinter.messagebox.showerror('登录提示', '用户名或密码错误！')
        else:  # 错误过多
            tkinter.messagebox.showerror('登录提示', '已连续错误5次！\n请稍后再试！')
            root.quit()  # 退出窗口





#----------------frame1--------------------
def dataframe_to_treeview(dfs, x1, y1, w, h, column_name='序号'):
    # 1.获取数据的列标题
    a = dfs.columns.values.tolist()
    a.insert(0, column_name)
    # 添加一个宽度列表，组成字典
    b = [80 for nums in range(len(a) - 1)]
    b.insert(0, 50)
    df_titles = dict(zip(a, b))
    # 2.设置纵向滚动条
    xbar = tkinter.Scrollbar(frame1, orient='horizontal')
    xbar.place(x=x1, y=y1 + h - 3, width=w)
    ybar = tkinter.Scrollbar(frame1, orient='vertical')
    ybar.place(x=x1 + w - 3, y=y1, height=h)
    # 3.创建Treeview
    tree = tkinter.ttk.Treeview(frame1, show='headings',
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
    '''
    龙虎榜功能
    :return: 在frame1画表
    '''
    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = time_temp.strftime('%Y%m%d')
    try:
        stock_lhb_detail_daily_sina_df = ak.stock_lhb_detail_daily_sina(date='end_date')
        df = stock_lhb_detail_daily_sina_df.iloc[:, 1:]
        df_rise = df[df['指标'] == '涨幅偏离值达7%的证券'].iloc[:, :6]
        df_fall = df[df['指标'] == '跌幅偏离值达7%的证券'].iloc[:, :6]
        label_rise = tkinter.Label(frame1, text='涨幅偏离值达7%的证券')
        label_rise.place(x=180, y=10)
        dataframe_to_treeview(df_rise, 180, 30, 600, 200)
        label_fall = tkinter.Label(frame1, text='跌幅偏离值达7%的证券')
        label_fall.place(x=180, y=270)
        dataframe_to_treeview(df_fall, 180, 290, 600, 200)
    except KeyError as e:
        status_str.set('休市，没有龙虎榜数据！')




def show_frame1():
    frame1.pack()
    stock_lhb()
    frame2.pack_forget()
    frame3.pack_forget()
    frame4.pack_forget()


#----------------frame2--------------------
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


def show_frame2():
    frame2.pack()
    label_code = tkinter.Label(frame2, text='股票代码：')
    label_code.place(x=20, y=20)
    code_example = tkinter.StringVar()
    code_example.set('sz000001')
    entry_code = tkinter.Entry(frame2, textvariable=code_example, width=30)
    entry_code.place(x=80, y=20)
    label_start_date = tkinter.Label(frame2, text='查询的开始日期：')
    label_start_date.place(x=400, y=20)
    start_date_example = tkinter.StringVar()
    start_date_example.set('19910403')
    entry_start_date = tkinter.Entry(frame2, textvariable=start_date_example, width=30)
    entry_start_date.place(x=500, y=20)

    # 获取数据
    def deliever_code_date():
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        status_str.set('获取历史数据成功！'+'获取该股票数据'+str(len(df))+'条！')

    button_get_data = tkinter.Button(frame2, text='开始获取',command=deliever_code_date)
    button_get_data.place(x=800, y=15)
    #分割线
    sep1 = tkinter.ttk.Separator(frame2, orient='horizontal')
    sep1.place(x=20, y=55, width=920, height=1)
    # 建库
    def deliever_code_dbpath():
        code = entry_code.get()
        dbpath = 'stock.db'
        status_str.set('存入数据中，请耐心等待--------')
        init_stockdb(code,dbpath)
        status_str.set('创建数据库成功！')

    button_createdb = tkinter.Button(frame2, text='创建数据库',command=deliever_code_dbpath,height=5, width=20, font='helv36')
    button_createdb.place(x=160, y=110)

    # 存数据
    def deliever_code_df_dbpath():
        code = entry_code.get()
        start_date = entry_start_date.get()
        dbpath = 'stock.db'
        df = get_history_datas(code, start_date)
        save_history_db(code,df,dbpath)
        status_str.set('成功存入数据'+str(len(df))+'条！')

    button_savedb = tkinter.Button(frame2, text='存入历史数据', command=deliever_code_df_dbpath,height=5,width=20,font='helv36')
    button_savedb.place(x=550, y=110)

    frame1.pack_forget()
    frame3.pack_forget()
    frame4.pack_forget()


#----------------frame3--------------------
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
                    # 模型存储路径 评价(平均误差率) 期数 记忆天数  预测天数 lstm层数 全连接层数 神经元数(.keras   .weights.h5)
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


def model_loading(df,modelpath,funtioncode):
    best_model = load_model(modelpath)
    df = df
    mday = re.findall('(?<=mem_)\d+', modelpath) #从文件名匹配记忆天数
    the_mem_days = int(mday[0])
    pday = re.findall('(?<=pre_)\d+', modelpath)  # 从文件名匹配预测天数
    pre_days = int(pday[0])
    X, y, X_lately = Stock_Price_LSTM_Data_Precesing(df, the_mem_days, pre_days)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # best_model.evaluate(X_test, y_test)
    pre = best_model.predict(X_test)
    if funtioncode == '1':
        y_fact_recent = y_test[-365:]
        y_pre_recent = pre[-365:]
        df_time = df.iloc[-len(y_fact_recent):, 0]
        plt.figure(dpi=300, figsize=(24, 8))
        plt.plot(df_time, y_fact_recent, color='red', label='price')
        plt.plot(df_time, y_pre_recent, color='green', label='predict')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        # plt.gcf().autofmt_xdate()
        plt.show()
    elif funtioncode == '2':
        y_fact_recent = y_test[-20:]
        y_pre_recent = pre[-20:]
        df_time = df.iloc[-len(y_fact_recent):, 0]
        plt.figure(dpi=300, figsize=(24, 8))
        plt.plot(df_time, y_fact_recent, color='red', label='price')
        plt.plot(df_time, y_pre_recent, color='green', label='predict')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # plt.gcf().autofmt_xdate()
        plt.show()


def show_frame3():
    frame3.pack()
    label_code = tkinter.Label(frame3, text='股票代码：')
    label_code.place(x=20, y=20)
    code_example = tkinter.StringVar()
    code_example.set('sz000001')
    entry_code = tkinter.Entry(frame3, textvariable=code_example, width=30)
    entry_code.place(x=80, y=20)
    label_start_date = tkinter.Label(frame3, text='查询的开始日期：')
    label_start_date.place(x=400, y=20)
    start_date_example = tkinter.StringVar()
    start_date_example.set('19910403')
    entry_start_date = tkinter.Entry(frame3, textvariable=start_date_example, width=30)
    entry_start_date.place(x=500, y=20)
    # 获取数据
    def deliever_code_date():
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        status_str.set('获取历史数据成功！' + '获取该股票数据' + str(len(df)) + '条！')

    button_get_data = tkinter.Button(frame3, text='开始获取', command=deliever_code_date)
    button_get_data.place(x=800, y=15)

    # 分割线
    sep1 = tkinter.ttk.Separator(frame3, orient='horizontal')
    sep1.place(x=20, y=55, width=920, height=1)

    label_title = tkinter.Label(frame3, text='基于LSTM的训练', font='helv36')
    label_title.place(x=400, y=70)

    label_mem_days = tkinter.Label(frame3, text='记忆天数：')
    label_mem_days.place(x=80, y=120)
    mem_days_example = tkinter.StringVar()
    mem_days_example.set('10,15')
    entry_mem_days = tkinter.Entry(frame3, textvariable=mem_days_example, width=30)
    entry_mem_days.place(x=150, y=120)

    label_pre_days = tkinter.Label(frame3, text='预测天数：')
    label_pre_days.place(x=400, y=120)
    pre_days_example = tkinter.StringVar()
    pre_days_example.set('10')
    entry_pre_days = tkinter.Entry(frame3, textvariable=pre_days_example, width=30)
    entry_pre_days.place(x=470, y=120)

    label_lstm_layers = tkinter.Label(frame3, text='lstm层数：')
    label_lstm_layers.place(x=30, y=180)
    lstm_layers_example = tkinter.StringVar()
    lstm_layers_example.set('1,2')
    entry_lstm_layers = tkinter.Entry(frame3, textvariable=lstm_layers_example, width=30)
    entry_lstm_layers.place(x=90, y=180)

    label_dense_layers = tkinter.Label(frame3, text='全连接层数：')
    label_dense_layers.place(x=270, y=180)
    dense_layers_example = tkinter.StringVar()
    dense_layers_example.set('1,2')
    entry_dense_layers = tkinter.Entry(frame3, textvariable=dense_layers_example, width=30)
    entry_dense_layers.place(x=350, y=180)

    label_units = tkinter.Label(frame3, text='神经元数：')
    label_units.place(x=530, y=180)
    units_example = tkinter.StringVar()
    units_example.set('16,32')
    entry_units = tkinter.Entry(frame3, textvariable=units_example, width=30)
    entry_units.place(x=600, y=180)
    # 数据处理，模型训练
    def deliever():
        status_str.set('正在数据预处理以及建立模型中，请耐心等待------')
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        me = entry_mem_days.get()
        mem_days = [int(t) for t in me.split(",")]
        pre_days = int(entry_pre_days.get())
        lst = entry_lstm_layers.get()
        lstm_layers = [int(t) for t in lst.split(",")]
        den = entry_dense_layers.get()
        dense_layers = [int(t) for t in den.split(",")]
        un = entry_units.get()
        units = [int(t) for t in un.split(",")]
        train_LSTM_model(df, mem_days, pre_days, lstm_layers, dense_layers, units)
        status_str.set('成功训练模型，请到models文件夹下查看------')

    button_train_models = tkinter.Button(frame3, text='开始训练生成',command=deliever)
    button_train_models.place(x=800, y=115)

    # 分割线  导入模型并预测
    sep2 = tkinter.ttk.Separator(frame3, orient='horizontal')
    sep2.place(x=20, y=250, width=920, height=1)

    label_title2 = tkinter.Label(frame3, text='导入模型生成预测推荐图', font='helv36')
    label_title2.place(x=370, y=260)

    # 路径选择
    label_content = tkinter.Label(frame3, text='选择目录：', font=('华文彩云', 15))
    label_content.place(x=200, y=320)
    # 输入框控件
    entry_text = tkinter.StringVar()
    entry_content = tkinter.Entry(frame3, textvariable=entry_text, font=('FangSong', 10), width=50, state='readonly')
    entry_content.place(x=300, y=325)

    # 按钮控件
    def get_path():
        """注意，以下列出的方法都是返回字符串而不是数据流"""
        # 返回一个字符串，且只能获取文件夹路径，不能获取文件的路径。
        # path = filedialog.askdirectory(title='请选择一个目录')

        # 返回一个字符串，可以获取到任意文件的路径。
        path = filedialog.askopenfilename(title='请选择文件')

        # 生成保存文件的对话框， 选择的是一个文件而不是一个文件夹，返回一个字符串。
        # path = filedialog.asksaveasfilename(title='请输入保存的路径')

        entry_text.set(path)

    button_path = tkinter.Button(frame3, text='选择路径', command=get_path)
    button_path.place(x=670, y=315)

    # 单选按钮
    funtion_code = tkinter.StringVar()
    funtion_code.set('1')
    radio = tkinter.Radiobutton(frame3, variable=funtion_code, value='1', text='近一年对比预测')
    radio.place(x=280, y=370)
    radio = tkinter.Radiobutton(frame3, variable=funtion_code, value='2', text='近二十天对比预测')
    radio.place(x=540, y=370)

    def deliever_df_path_funtioncode():
        status_str.set('导入模型中-------')
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        modelpath = entry_text.get()
        funtioncode = funtion_code.get()
        model_loading(df,modelpath,funtioncode)
        status_str.set('预测成功！')

    button_path = tkinter.Button(frame3, text='生成对比预测推荐图',command=deliever_df_path_funtioncode)
    button_path.place(x=420, y=430)

    frame1.pack_forget()
    frame2.pack_forget()
    frame4.pack_forget()


#----------------frame4--------------------
def stock_trend(df,funtioncode1):
    '''
    股票趋势图功能
    :param df: dataframe数据对象  date open high low close volume amount outstanding_share turnover
    :param funtioncode: 功能码1看历史全部，2近十年，3近十二个月，4近一个季度，5近一个月，6近七天
    :return:
    '''
    plt.figure(dpi=300, figsize=(24, 8))
    if funtioncode1 == '1':#看历史全部
        df_y_close = df['close'].values
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.show()
    elif funtioncode1 == '2':#近十年
        df_y_close = df['close'].values[-3650:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.show()
    elif funtioncode1 == '3':#近十二个月
        df_y_close = df['close'].values[-365:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.show()
    elif funtioncode1 == '4':#近一个季度
        df_y_close = df['close'].values[-90:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.show()
    elif funtioncode1 == '5':#近30天
        df_y_close = df['close'].values[-30:]
        df_X_time = df.iloc[-len(df_y_close):, 0]
        plt.plot(df_X_time, df_y_close, color='red', label='price')
        plt.xticks(rotation=90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.show()
    elif funtioncode1 == '6':#近七天
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
    if funtioncode2 == '1':
        mpf.plot(df_cleaned.iloc[-30:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近30天K线图（均线5日、10日）',
                 mav=(5, 10),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")
    elif funtioncode2 == '2':
        mpf.plot(df_cleaned.iloc[-90:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近90天K线图（均线5日、10日）',
                 mav=(5, 10),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")
    elif funtioncode2 == '3':
        mpf.plot(df_cleaned.iloc[-365:],
                 type='candle',
                 ylabel="price",
                 style=my_style,
                 title='所选股票近1年K线图（均线20日、60日）',
                 mav=(20, 60),
                 volume=True,
                 figratio=(5, 3),
                 ylabel_lower="Volume")


def show_frame4():
    frame4.pack()
    label_code = tkinter.Label(frame4, text='股票代码：')
    label_code.place(x=20, y=20)
    code_example = tkinter.StringVar()
    code_example.set('sz000001')
    entry_code = tkinter.Entry(frame4, textvariable=code_example, width=30)
    entry_code.place(x=80, y=20)
    label_start_date = tkinter.Label(frame4, text='查询的开始日期：')
    label_start_date.place(x=400, y=20)
    start_date_example = tkinter.StringVar()
    start_date_example.set('19910403')
    entry_start_date = tkinter.Entry(frame4, textvariable=start_date_example, width=30)
    entry_start_date.place(x=500, y=20)
    # 获取数据
    def deliever_code_date():
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        status_str.set('获取历史数据成功！' + '获取该股票数据' + str(len(df)) + '条！')

    button_get_data = tkinter.Button(frame4, text='开始获取', command=deliever_code_date)
    button_get_data.place(x=800, y=15)
    # 分割线
    sep1 = tkinter.ttk.Separator(frame4, orient='horizontal')
    sep1.place(x=20, y=55, width=920, height=1)

    label_title = tkinter.Label(frame4, text='股票趋势', font='helv36')
    label_title.place(x=400, y=70)

    funtion_code1 = tkinter.StringVar()
    funtion_code1.set('1')
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='1', text='历史全部')
    radio.place(x=80, y=110)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='2', text='近十年')
    radio.place(x=200, y=110)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='3', text='近十二个月')
    radio.place(x=320, y=110)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='4', text='近一个季度')
    radio.place(x=440, y=110)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='5', text='近一个月')
    radio.place(x=560, y=110)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code1, value='6', text='近七天')
    radio.place(x=680, y=110)

    def deliever_df_funtioncode1():
        status_str.set('画图中-------')
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        funtioncode1 = funtion_code1.get()
        stock_trend(df,funtioncode1)
        status_str.set('趋势图')

    button_draw1 = tkinter.Button(frame4, text='查看', height=2, width=7, font='helv36',command=deliever_df_funtioncode1)
    button_draw1.place(x=400, y=160)

    # 分割线
    sep2 = tkinter.ttk.Separator(frame4, orient='horizontal')
    sep2.place(x=20, y=250, width=920, height=1)

    label_title = tkinter.Label(frame4, text='股票K线', font='helv36')
    label_title.place(x=400, y=270)

    funtion_code2 = tkinter.StringVar()
    funtion_code2.set('1')
    radio = tkinter.Radiobutton(frame4, variable=funtion_code2, value='1', text='近30天')
    radio.place(x=200, y=320)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code2, value='2', text='近90天')
    radio.place(x=400, y=320)
    radio = tkinter.Radiobutton(frame4, variable=funtion_code2, value='3', text='近1年')
    radio.place(x=600, y=320)

    def deliever_df_funtioncode2():
        status_str.set('画图中-------')
        code = entry_code.get()
        start_date = entry_start_date.get()
        df = get_history_datas(code, start_date)
        funtioncode2 = funtion_code2.get()
        kline(df,funtioncode2)
        status_str.set('K线图')

    button_draw2 = tkinter.Button(frame4, text='查看', height=2, width=7, font='helv36',command=deliever_df_funtioncode2)
    button_draw2.place(x=400, y=370)

    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()


# def show_frame5():    菜单备选
#     frame5.pack()
#     frame1.pack_forget()
#     frame2.pack_forget()
#     frame3.pack_forget()
#     frame4.pack_forget()
#     frame6.pack_forget()
#     frame7.pack_forget()






if __name__ == '__main__':
    root = tkinter.Tk()  # 创建主窗口
    root.title('主窗口')  # 主窗口标题
    root.geometry('0x0')  # 设置主窗口大小为 0
    root.overrideredirect(True)  # 暂时隐藏主窗口外框

    login = tkinter.Toplevel()  # 创建登录窗口
    login.title('用户登录')  # 登录窗口的标题
    login.geometry('250x200+500+250')  # 登录窗口的大小及位置
    login.resizable(False, False)  # 设置登录窗口的大小不可改变

    login.protocol('WM_DELETE_WINDOW', root.quit)  # 关闭Toplevel的同时，关闭主窗口

    image = tkinter.PhotoImage(file='res/bg%s.png' % random.randint(0, 6))  # 随机选取一个图片
    tkinter.Label(login, image=image, bd=0, text='登录窗口\n股票预测系统', compound='center', font=('华文行楷', 25),
                  fg='yellow').place(width=250, height=100)  # 创建一个图片标签
    tkinter.Label(login, text='用户').place(width=50, height=25, x=20, y=105)  # “用户”文字标签
    tkinter.Label(login, text='密码').place(width=50, height=25, x=20, y=135)  # “密码”文字标签
    (account := tkinter.ttk.Entry(login)).place(width=160, height=25, x=70, y=105)  # 用户名输入框
    (password := tkinter.ttk.Entry(login, show='●')).place(width=160, height=25, x=70, y=135)  # 密码输入框
    tkinter.ttk.Button(login, text='注册', command=lambda: toplevel_register()).place(width=100, height=28, x=20,
                                                                                    y=166)  # 注册按钮
    tkinter.ttk.Button(login, text='登录', command=lambda: test_for_password()).place(width=100, height=28, x=130,
                                                                                    y=166)  # 登录按钮

    # 根界面上的框架
    frame1 = tkinter.ttk.Frame(root, width=960, height=500)
    frame1.pack()
    frame2 = tkinter.ttk.Frame(root, width=960, height=500)
    frame2.pack()
    frame3 = tkinter.ttk.Frame(root, width=960, height=500)
    frame3.pack()
    frame4 = tkinter.ttk.Frame(root, width=960, height=500)
    frame4.pack()
    # frame5 = tkinter.ttk.Frame(root, width=960, height=500)   菜单备选
    # frame5.pack()


    # 根界面上菜单
    menubar = tkinter.Menu(root)
    menubar.add_command(label='龙虎榜', command=show_frame1)
    menubar.add_command(label='股票历史数据导出', command=show_frame2)
    menubar.add_command(label='自定义模型训练生成与导入及预测', command=show_frame3)
    menubar.add_command(label='股票可视化与分析', command=show_frame4)
    # menubar.add_command(label='导入模型并预测出图', command=show_frame5)  # 菜单备选
    root.config(menu=menubar)

    # 根界面分割线和状态栏
    separator = tkinter.ttk.Separator(root, orient='horizontal')
    separator.place(x=0,y=501,width=960,height=1)
    status_str = tkinter.StringVar()
    status_str.set('这是状态栏')
    status_message = tkinter.Message(root, textvariable=status_str, width=960)
    status_message.place(x=0, y=510)

    root.title('机器学习股票预测推荐系统v1.0')
    root.mainloop()  # 窗口循环
