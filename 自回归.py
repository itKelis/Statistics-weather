import numpy as np
import pandas as pd

t = np.array([2.5,2.4,2.6,2.3,2.5,2.4,2.8,3.0,2.4,2.1,2.9,3.1])
# t = np.array([2,1,3,2,0,4,3,-1,2,5,0,3])
mean = np.mean(t)
S = np.var(t)
print(mean)
print(t[11]-mean)
print(t[9]-mean)
print(t[8]-mean)
def tao(t):
    #i就是tao
    rou = np.zeros((t.shape[0],int(t.shape[0]/2+1)))
    for i in range(t.shape[0]):
        for j in range(int(t.shape[0]/2)+1):
            if ((i+j)<t.shape[0]):
                rou[i][j] = ((t[i]-mean)*(t[i+j]-mean))
    summarize = np.sum(rou,axis=0)
    print(summarize)
    for i in range(summarize.shape[0]):
        summarize[i] = (summarize[i]/(12-i))
    return summarize/S
print(tao(t))

