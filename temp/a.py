import numpy as np
import pandas as pd

a = np.array(pd.read_csv("./temp/1.csv"))
b = np.array(a.T[0:-1].T)
# print(b)S
c = np.array([0.1373735,0.1487221,-0.280174,0.2131978])
print(np.linalg.solve(np.array(b),c))