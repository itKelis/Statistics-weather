import numpy as np
import pandas as pd

class multi():
    def __init__(self,file,sheet_name,start,size,select = None) -> None:
        #start 代表起始年份
        #size 代表往后提取年份的数量
        #stop 代表提取多少行，年份一一对应
        #sheet_name 代表要读取哪个表

        #读取所有表的数据
        data = pd.read_excel(file,sheet_name=sheet_name)
        
        #计算x1,x2,x3,x4
        procceed = self.dataprocess(data,sheet_name,size,select)
        self.datas = procceed[:-1]
        self.y = procceed[-1]

    def countL(self,x,y):
        #计算l
        return ((np.sum(x*y))-(1/(x.shape[0]-1))*np.sum(x)*np.sum(y))

    def sloveRegression(self):
        #计算L矩阵并解算出估计方程
        #计算出b，b0，Ly，y估计
        l = np.zeros((4,4))

        for i in range(4):
            for j in range(4):
                # if j >= i :             
                # #i和j都是选择x几
                l[i][j] = self.countL(self.datas[i],self.datas[j])
        
        ly = np.zeros(4)
        for i in range(4):
            ly[i] = self.countL(self.datas[i], self.y)
        
        self.l = l
        self.ly = ly
        print(l)
        print(ly)
        self.b = np.linalg.solve(l, ly)     
        self.b0 = np.mean(self.y) - (np.mean(self.datas[0])*self.b[0]) 
        - (np.mean(self.datas[1])*self.b[1]) 
        - (np.mean(self.datas[2])*self.b[2]) 
        - (np.mean(self.datas[3])*self.b[3])

        self.ye = lambda x1,x2,x3,x4,b,b0:eval("{}*x1+{}*x2+{}*x3+{}*x4+{}".format("b[0]","b[1]","b[2]","b[3]","b0"))

    def checkRegression(self):
        #计算U,Q,Lyy,F,R
        Us = []
        for i in range(self.datas.shape[1]):
            Us.append((self.ye(self.datas[0][i],self.datas[1][i],self.datas[2][i],self.datas[3][i],self.b,self.b0) - np.mean(self.y))**2)
        self.U = np.sum(Us)

        Qs = []
        for i in range(self.datas.shape[1]):
            Qs.append((self.y[i] - self.ye(self.datas[0][i],self.datas[1][i],self.datas[2][i],self.datas[3][i],self.b,self.b0))**2)

        self.Q = np.sum(Qs)
        self.Lyy = self.U + self.Q
        self.F = self.U/self.Q/(self.datas.shape[1]-2)
        self.R = self.U/self.Lyy
    
    def printResult(self):
        print(f'''L矩阵是:\n{self.l}\n
        F是：{self.F}
        R是：{self.R}
        Lyy是：{self.Lyy}
        Q是：{self.Q}
        U是：{self.U}
        Ly是：{self.ly}
        ''')
        pass

    def dataprocess(self,data,sheet_name,size,select, dimension = None):
        a = []      
        #a用来保存处理前的数据
        
        if dimension == None:
            #读取每个表所需年份的所有数据
            #手动输入数据维度，一元则直接读取数据，多元则循环读取数据
            for i in range(len(sheet_name)):
                datas = np.array(data[sheet_name[i]])[:size+1,1:13]     
                a.append(datas)
        else:
            datas = np.array(data)[:size+1,1:13]     
            a.append(datas)

        #对每个表按所需月份做平均
        if select is not None:      
            #select是每个选择月份，如果是列表则每个表都提取相同年份，如果则每个表提取所指定的年份
            for i in range(len(sheet_name)):
                    b = np.transpose(a[i])
                    month = None

                    if isinstance(select,list):
                        #如果选择的月份以列表形式传入
                        #则用列表推导式读取对应月份数据
                        #以字典形式传入 
                        #则用列表推导式读取对应月份的值
                        month = np.array([b[j] for j in select])
                    elif isinstance(select,dict):
                        month = np.array([b[j] for j in select[sheet_name[i]]])
                    a[i] = np.mean(np.transpose(month),axis=1)
                    
        if dimension == None:
            return np.array(a)
        else:
            return np.array(a).flatten()
    
sheet_name = ["Arctic Oscillation","North Atlantic Oscillation","NINO","North Pacific index","Western Pacific index"]
months = {"Arctic Oscillation":[11,0,1],"North Atlantic Oscillation":[11,0,1],"NINO":[5,6,7],"North Pacific index":[2,3,4],"Western Pacific index":[5,6,7]}
# months = [1,4,5]

multis = multi("./data/train2.xlsx",sheet_name,1979,39,select = months)
multis.sloveRegression()
multis.checkRegression()
multis.printResult()