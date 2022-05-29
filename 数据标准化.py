import numpy as np
import math
import matplotlib.pyplot as plt

#问题：为何气象统计分析一般都是距平值或标准化变量
#答：
#1、各种气象数据的单位，参考的基准值一般都是不同的，通过标准化或距平计算之后，可以将各种不同的变量放在同一个水平线上相互比较异常程度
#2、经过标准化处理后变量值一般在-3到3之间，绘图方便
#3、距平达到或大于2倍方差概率不到5%，绘图时的梯度可以很小，图像更精确，比较更准确

vars = [ 0.9, 1.2, 2.2, 2.4, -0.5, 2.5, -1.1, 0, 6.2, 2.7]              #气象数据
mean = np.mean(vars)                                                    #求均值
StdD = math.sqrt(np.sum([(x-mean) ** 2 for x in vars ])/len(vars)-1)    #求均方差，小样本使用n-1
Xz =[]
for t in vars:
    Xz.append((t-mean)/StdD)                                            #求出Xz 
print(Xz)
#下面是根据Xz画图
#
index = list(range(1,len(vars)+1))
loc = plt.gca()
loc.spines["right"].set_visible(False)
loc.spines["top"].set_visible(False)
loc.spines["bottom"].set_position("center")
plt.ylim(-3,3)
plt.xticks(index)
plt.plot(index,Xz,marker="d")
for x,y in zip(index,Xz):
    if y>0:
        plt.text(x,y,format(y,".4f"),ha = "left",va = "bottom")
    else:
        plt.text(x,y,format(y,".4f"),ha = "left",va = "top")
plt.show()