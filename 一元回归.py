import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class simple_regression():
    #对象内变量解释
    #x：时间,数字从1开始
    #y: 做平均后的原始数据
    #b：估计方程的b
    #b0：估计方程b0
    #ye：y估计方程的字符串形式
    #yec：y估计方程的函数形式
    #U：回归平方和
    #Q：残差平方和
    #R：R值
    #F：F值
    #convince：置信度
    def __init__(self) -> None:
        #读取数据
        # datas = pd.DataFrame(pd.read_csv("一元回归.csv"),columns=["year","3","4","5"])

        datas = pd.read_excel("./data/train1.xlsx")
        self.y = self.dataprocess(datas,["sheet"],60,[10,8,9],dimension=True)
        self.x = np.array(range(1950,2010))
        # print(procceed)

        # print(datas)
        # self.x = datas.to_numpy()[:,0]-datas.to_numpy()[0][0]+1
        # print(self.x)
        # self.y = np.mean(datas.to_numpy()[:,1:],axis = 1)
        
    def estimate(self):                         #计算估计方程的b,b0和回归方程
        self.b = self.countb(self.x,self.y)
        self.b0 = np.mean(self.y)-(self.b * float(np.mean(self.x)))
        self.b0c = lambda x: "+{}".format(x) if x >= 0 else x
        self.ye = "y={}x{}".format(self.b,self.b0)
        self.yec = lambda x:eval("{}*x+{}".format(self.b,self.b0c(self.b0)))
    
    def check(self):                #计算U,Q,Lyy,F,R和置信区间
        self.U = sum((np.array([self.yec(i) for i in self.x])-np.mean(self.y))**2) 
        self.Q = sum((self.y - np.array([self.yec(i) for i in self.x]))**2)
        self.Lyy = self.U+self.Q
        self.F=(self.U/1)/(self.Q/(self.x.shape[0]-2))
        self.R = self.U/self.Lyy
        self.convince = (self.Q/(self.x.shape[0]-2))**2
    def printResult(self):
        print(f"""
            回归方程是：{self.ye}
            置信度是：{self.convince}
            R是：{self.R}
            F是：{self.F}
        """)
    def countb(self,x,y):
    #算出b
        b = ((np.sum(y*x))-(1/x.shape[0])*np.sum(x)*np.sum(y))/(np.sum([i**2 for i in x ])-(np.sum(x)/x.shape[0])**2)
        return b

    def plot(self):
        #y估计方程的头尾两点
        x = [self.x[0],self.x[-1]]
        y = [self.yec(x[0]), self.yec(x[1])]

        plt.scatter(self.x,self.y)          #y估计
        plt.plot(x,y)                       #原始数据散点图
        print("开始plot")
        plt.show()

    def dataprocess(self,data,sheet_name,size,select,dimension = None):
        a = []      #a用来保存处理前的数据
        
        #读取每个表所需年份的所有数据
        datas = np.array(data)[:size+1,1:13]     
        a.append(datas)
    
        #对每个表按所需月份做平均
        if select is not None:      #select是每个选择月份，如果是列表则每个表都提取相同年份，如果则每个表提取所指定的年份
            for i in range(len(sheet_name)):
                # for i in [1]:
                    b = np.transpose(a[i])
                    month = None
                    if isinstance(select,list):
                        month = np.array([b[j] for j in select])
                    elif isinstance(select,dict):
                        month = np.array([b[j] for j in select[sheet_name[i]]])
                    a[i] = np.mean(np.transpose(month),axis=1)
        if dimension == None:
            return np.array(a)
        else:
            return np.array(a).flatten()

sr = simple_regression()
sr.estimate()
sr.check()
sr.printResult()
sr.plot()
#画图
