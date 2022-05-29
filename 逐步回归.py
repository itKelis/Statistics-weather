import numpy as np 
import pandas as pd
n=13
m=4

class stepcol():
    def __init__(self):
        data = pd.DataFrame(pd.read_csv("111.csv"),columns=["x1","x2","x3","x4","y"])
        self.R = []
        self.R.append(data.corr().to_numpy())
        print(self.R)
        
    def countV(self,time):
        v = np.ones(self.R[time].shape[1]-1)
        for i in range(self.R[time].shape[1]-1):
            v[i] = (self.R[time][i][-1]**2)/self.R[time][i][i]
        max = np.where(v==np.max(v))[0][0]
        print("最大值v{},为{},其他分别为{}".format(max,v[max],v))
        f = self.countF(time,v[max])
        print("f为{}".format(f))
        return v

    def countF(self,time,vmax):
        F = vmax/(self.R[time][-1][-1])/(np.size(self.R[time]) - time-2)
        return F
        pass


stepcols = stepcol()
stepcols.countV(0)
#信度0.05,为4,n=13,m=4
# xmean = [0.3650,20.06,233.8850,16.06]
# R = np.array([[1,-0.0158,-0.2176,0.2888,-0.3019],
#              [-0.0158,1,0.4331,-0.2882,0.5128],
#              [-0.2176,0.2888,1,0.0034,0.5297],
#              [0.2888,-0.2882,0.034,1,0.1578],
#              [-0.3019,0.5128,0.5297,0.1578,1]])
# #第一步计算方差贡献
# def contribute(r):
#     v = []
#     for i in range(r.shape[0]-1):
#         print("{}**2/{}".format(r[i][-1],r[0][i]))
#         v.append((r[i][-1]**2)/r[i][i])
#     print(max(v))


# def convert_matrix(r,k):
#     col=r.shape[1]
#     k=k-1#从第零行开始计数
#     #第k行的元素单不属于k列的元素
#     r1 = np.ones((col, col))  # np.ones参数为一个元组(tuple)
#     for i in range(col):
#         for j in range(col):
#             if (i==k and j!=k):
#                 r1[i,j]=r[k,j]/r[k,k]
#             elif (i!=k and j!=k):
#                 r1[i,j]=r[i,j]-r[i,k]*r[k,j]/r[k,k]
#             elif (i!= k and j== k):
#                 r1[i,j] = -r[i,k]/r[k,k]
#             else:
#                 r1[i,j] = 1/r[k,k]
#     return r1

# print(convert_matrix(R,3))

    