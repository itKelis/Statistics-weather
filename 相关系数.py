import numpy as np

class correlationProcess:
    def __init__(self,x,y) -> None:
        self.X = x
        self.y = y

    def Pearson(self,x = None,y = None):
        if not(x and y):
            x = self.X
            y = self.y
        self.n = len(x)
        E = lambda x:np.mean(x)
        S = lambda x:np.std(x)
        r = E(E((x - E(x)) * (y - E(y))) 
                            / (S(y) * S(x)) )   #利用矩阵运算算出相关系数
        

        self.r = r
        print("得到的皮尔森系数为{}\n".format(self.r))
    
    def PearsonCheck(self):
        if(self.r is not None):
            t = abs(self.r * np.sqrt((self.n-2)/(1-self.r ** 2)))
            print("自由度为：{}".format(self.n-2))
            print("显著性检验得到的t为{}".format(t))
        else:
            print("请先执行Pearson()计算相关系数再检验")

Xs = np.array([
    [4.1, 5.3, 4.3, 1.2, 3.4, 4.8, 3.9, 2.1, 2.3, -1.7, 2.0, 2.2],
    [3.9, 5.9, 7.7, 8.8, 4.0, 5.0, 2.3, 8.0, 2.9, 10.5, 8.5, 7.0],
    [13.0, 4.6, 2.8, 1.4, 7.1, 10.4, 13.0, 4.2, 8.7, 1.5, 3.6, 4.8]])

count = correlationProcess(Xs[2],np.mean(Xs,axis=0))
count.Pearson()
count.PearsonCheck()

def pearson(x,y):
    r = np.mean(np.mean((x - np.mean(x)) * (y - np.mean(y))) 
                            / (np.std(y) * np.std(x)) )   #利用矩阵运算算出相关系数
    return r