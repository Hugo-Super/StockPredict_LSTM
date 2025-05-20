import tkinter as tk

def creat_windows():
    win = tk.Tk()  # 创建窗口
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 800, 450
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('LSTM股票预测')  # 窗口命名

    #f_open =open('dataset_2.csv')
    canvas = tk.Label(win)
    canvas.pack()

    var = tk.StringVar()  # 创建变量文字
    var.set('选择数据集')
    tk.Label(win, textvariable=var, bg='#C1FFC1', font=('宋体', 21), width=20, height=2).pack()

    tk.Button(win, text='选择数据集', width=20, height=2, bg='#FF8C00', command=lambda: getdata(var, canvas),
              font=('圆体', 10)).pack()

    canvas = tk.Label(win)
    L1 = tk.Label(win, text="选择你需要的 列(请用空格隔开，从0开始）")
    L1.pack()
    E1 = tk.Entry(win, bd=5)
    E1.pack()
    button1 = tk.Button(win, text="提交", command=lambda: getLable(E1))
    button1.pack()
    canvas.pack()
    win.mainloop()

def getdata(var, canvas):
    pass

def getLable(E1):
    pass


if __name__ == "__main__":
    creat_windows()