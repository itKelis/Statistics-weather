from optparse import check_builtin
import numpy as np
import pandas as pd

class discrimination():
    def __init__(self,checks) -> None:
        sheet_name = ["1","2"]
        select = {"1" : [0,1,2,3],"2":[0,1,2,3]}
        size = 20
        data = pd.read_excel("./data/train3.xlsx",sheet_name=["1","2"])
        # print()
        # procceed = self.dataprocess(data,sheet_name,size,select,dimension = 1)
        # print(procceed.shape)
        self.x = np.array([np.array(data["1"])[:19,1:],np.array(data["2"])[:19,1:]])
        print(checks)
        self.checkdata = checks
        # self.y = np.array([[],[],[],[]])
        pass
    
    def countX(self):
        #[[第一类],[第二类]]
        #类里面[[第一因子],[第二因子],[第三因子]]
        # print(self.x[0].shape)
        x = np.array([self.x[0].T,self.x[1].T])
        print(x[0].shape)
        h = x[0].shape[0]
        l = x[0].shape[1]
        Ws = np.zeros((h,h))

        for i in range(h):
            for j in range(h):
                Ws[i][j] = (np.sum(np.array([k - np.mean(x[0][i]) for k in x[0][i]]) * np.array([k - np.mean(x[0][j]) for k in x[0][j]]))
                + np.sum(np.array([k - np.mean(x[1][i]) for k in x[1][i]]) * np.array([k - np.mean(x[1][j]) for k in x[1][j]])))

        self.d = np.zeros(h)
        for i in range(h):
            self.d[i] = np.mean(x[0][i] - x[1][i])

        self.W = np.linalg.solve(Ws,self.d)
        self.yce = eval("lambda x : x[0] * {} + x[1] * {} + x[2] * {} + x[3] * {}".format(self.W[0],self.W[1],self.W[2],self.W[3]))  
        print(np.mean(x[0], axis= 1)) 
        self.y1 = self.yce(np.mean(x[0], axis= 1))
        self.y2 = self.yce(np.mean(x[1], axis= 1)) 
        self.yc =  (((self.x[0].size * self.y1) +(self.x[1].size * self.y2))/
                (int(self.x[0].size) + int(self.x[1].size)))
        pass

    def check(self):
        # print()
        nums = np.zeros(len(self.checkdata))
        for i in range(len(self.checkdata)):
            nums[i] = self.yce(self.checkdata[i])
        self.checks = nums
        # print(self.x[0].shape[0])
        
        y1str = "属于海雾"
        y2str = "属于轻雾"
        for j in range(len(self.checks)):
            if self.y1 > self.y2:
                if self.checks[j] > self.yc:
                    print("第{}个{}".format(j,y1str,self.checkdata))
                else:
                    print("第{}个{}".format(j,y2str))                

    def inspect(self):
        # ck = np.mean(self.x[0],axis=0) - np.mean(self.x[0],axis=0)
        # print(self.W)
        self.D = ((self.x.size-2) * np.sum(self.W*self.d))
        self.F = (((self.x[1].size*self.x[0].size)/self.x.size) * ((self.x.size-self.W.size-1)/((self.x.size-2) * self.W.size)) * self.D)
        print("D是{}\nF是{}".format(self.D,self.F))

    def dataprocess(self,data,sheet_name    ,size,select, dimension = None):
        a = []      
        #a用来保存处理前的数据
        
        if dimension == None:
            #读取每个表所需年份的所有数据
            #手动输入数据维度， 一元则直接读取数据，多元则循环读取数据
            for i in range(len(sheet_name)):
                datas = np.array(data[sheet_name[i]])[:size+1,1:13]     
                print("dimension")
                a.append(datas)
        else:
            datas = np.array(data)[:size+1,1:13]     
            a.append(datas)
        print("a是:{}".format(a))

        #对每个表按所需月份做平均
        if select is not None:      
            #select是每个选择月份，如果是列表则每个表都提取相同年份，如果则每个表提取所指定的年份
            for i in range(len(sheet_name)):
                    b = np.transpose(a[i])
                    print(b)
                    month = None

                    if isinstance(select,list):
                        #如果选择的月份以列表形式传入
                        #则用列表推导式读取对应月份数据
                        #以字典形式传入 
                        #则用列表推导式读取对应月份的值
                        month = np.array([b[j] for j in select])
                    elif isinstance(select,dict):
                        month = np.array([b[j] for j in select[sheet_name[i]]])
                    a[i] = np.transpose(month)
                    
        if dimension == None:
            return np.array(a,dtype='object')
        else:
            return np.array(a,dtype='object').flatten()
            
checks = [[300,85,0.34,3.4],[283,88,-0.18,4.1],[200,78,1.2,4.32],[268,94,2.31,2.0],[180,96,1.5,3.62]]   #检验的数据
a = discrimination(checks)
a.countX()
a.check()
a.inspect()