import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class timeanaliize():
    def __init__(self, size):
        self.data = pd.read_excel("./data/for.xlsx")
        # print(data.to_numpy())
        x = self.data.to_numpy()[:,0][:size+1]
        self.x = x - x[0]+1
        self.y = self.data.to_numpy()[:,1][:size+1]
        self.size = size
        # print(x)
    
    def Xdt(self):
        n = len(self.x)
        Xd = np.zeros(int(n/2))
        # print(self.x)
        for t in range(int(n/2)):
            xts = 0
            for j in range(len(self.x)):
                if t == 0:
                    # print(self.y[j] - np.mean(self.y))
                    xts += pow((self.y[j] - np.mean(self.y)),2)
                    # Xd[t] = (self.x - np.concatenate(np.sqrt(np.mean(self.x) )))/n
                else:
                    if (j + t) < n:
                        xts += ((self.y[j] - np.mean(self.y)) * (self.y[j+t] - np.mean(self.y)))/Xd[0]
            Xd[t] = xts /(n-t)
        self.rou = Xd[1:]
        print("rou是{}".format(len(self.rou)))
        max = self.FindList3MaxNum(5)
        self.result = np.zeros(len(max))
        #打印
        for i in range(len(max)):
            self.result[i] = np.where(self.rou == max[i])[0]

    def slove(self):
        so = np.zeros((len(self.result),len(self.result)))
        self.d = np.zeros((len(self.result),len(self.result)))

        for i in range(len(self.result)):
            for j in range(len(self.result)):
                so[i][j] = self.rou[i-j]
                self.d[i][j] = i-j
                
        f = np.array([self.rou[int(i)] for i in self.result])
        self.fi = np.linalg.solve(np.array(so), f)
        print("fi是{}".format(self.fi))
        self.Xdc =eval("lambda x : x[0]*{}+x[1]*{}+x[2]*{}+x[3]*{}".format(self.fi[0],self.fi[1],self.fi[2],self.fi[3])) #
        
    def check(self):
        x = self.data.to_numpy()[:,1][:]
        xguji = np.zeros(len(x))
        print(range(self.size,len(x)))
        print(len(x))
        self.xguji = xguji
        for i in range(self.size,len(x)):
            print(i)
            xguji[i] = self.Xdc(np.array([x[i-6]-np.mean(x[:i+1]),x[i-11]-np.mean(x[:i+1]),x[i-4]-np.mean(x[:i+1]),x[i-3]-np.mean(x[:i+1]),x[i-7]-np.mean(x[:i+1])]))
        print(xguji)

    def plot():
        
        pass
    def FindList3MaxNum(self,size):  # 快速获取list中最大的五个元素
        # max1, max2, max3, max4, max5 = None, None, None, None,None
        max = np.zeros(size)
        for num in self.rou:
            for i in range(len(max)):
                if abs(num) > max[i]:
                    max[i], num = num, max[i]

        return max
            # if max1 is None or max1 < num:
            #     max1, num = num, max1
            # if num is None:
            #     continue
            # if max2 is None or num > max2:
            #     max2, num = num, max2
            # if num is None:
            #     continue
            # if max3 is None or num > max3:
            #     max3, num = num, max3
            # if num is None:
            #     continue
            # if max4 is None or num > max4:
            #     max4, num = num, max4
            # if num is None:
            #     continue
            # if max5 is None or num > max5:
            #     max5, num = num, max5    
        

size = 25
a = timeanaliize(size)
a.Xdt()
a.slove()
a.check()
# a = [[1,-0.041,-0.015,0.195],[-0.041,1,-0.041,-0.074],[-0.015,-0.041,1,0.142],[0.195,0.074,0.141,1]]
# b = [0.195,0.149,-0.257,-0.247]
# print(np.linalg.solve(a, b))