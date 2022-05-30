import numpy as np
import pandas as pd

class discrimination():
    def __init__(self) -> None:
        sheet_name = ["1","2"]
        select = {"1" : [0,1,2,3],"2":[0,1,2,3]}
        size = 20
        data = pd.read_excel("./data/train3.xlsx",sheet_name=["1","2"])
        # print()
        # procceed = self.dataprocess(data,sheet_name,size,select,dimension = 1)
        # print(procceed.shape)
        self.x = np.array([np.array(data["1"])[0:19,1:],np.array(data["2"])[0:19,1:]])
        self.y = None
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
        # print(Ws)
        
        self.d = np.zeros(h)
        for i in range(h):
            self.d[i] = np.mean(x[0][i] - x[1][i])

        self.W = np.linalg.solve(Ws,d)
        self.yc = (np.size(self.x[0])*np.mean(self.y)
            )
        
        pass

    def dataprocess(self,data,sheet_name,size,select, dimension = None):
        a = []      
        #a用来保存处理前的数据
        
        if dimension == None:
            #读取每个表所需年份的所有数据
            #手动输入数据维度，一元则直接读取数据，多元则循环读取数据
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

a = discrimination()
a.countX()