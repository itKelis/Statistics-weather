import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

class DataProcess:
    def __init__(self,sta = "CHM00059287",step = [5,10,15],mark = 0.01, marked = 3.2,Dtype = "year",column = "TAVG") -> None:
        self.columns = column.split()
        self.step = step
        self.marked = marked
        self.mark = mark
        self.type = Dtype
        self.sta = sta
        if(self.getYearsum()):
            self.year,self.temp = self.ReadData()
        else:
            os._exit(1)
    
    #函数名：Tprocess
    #注释：对数据进行滑动t检验并生成图表
    def Tprocess(self):
        mutations = {}
        for step in self.step:
            years = self.year[step:len(self.year)-step]
            t = self.tslide(step)
            self.tplot(t,step)
            mutation = []
            for j in range(len(t)):
                if(abs(t[j])>self.marked):
                    mutation.append(years[j])
            mutations[step] = mutation
        print("滑动T检验突变年份：{}".format(mutations))
        return mutations
    
    #函数名：MKprocess
    #注释：对数据进行MK检验并生成图表
    def MKprocess(self):
        UFk = self.MKcount(self.temp)
        UBk = self.MKcount(self.temp[::-1],True)
        self.MKplot(UFk = UFk, UBk = UBk)
        distance = 100.0
        for i in range(len(UFk)):
            if(abs(UFk[i] - UBk[i]) < distance):
                distance = i
        print("MK检验突变开始年份大概为：{}".format(self.year[distance]))

    #函数名：tslide
    #参数：temp，step
    #注释：temp为年平均气温数据，step为滑动步长,通过循环对公式计算
    def tslide(self,step):
        n = np.sqrt((1/step)+(1/step))          #公式中分母一部分
        length = len(self.temp) 
        t = np.zeros(length - 2 * step )        #初始化空数组保存计算的到的t
        # t = np.zeros(length ) 
        for i in range(step-1,length-step):
            x1 = self.temp[i-step+1:i+1]
            x2 = self.temp[i+1:i+step+1]
            x1_mean = np.mean(x1)   
            x2_mean = np.mean(x2)
            s1 = np.var(x1)         
            s2 = np.var(x2)
            s = np.sqrt((step * s1 + step * s2  )/(step+step-2))
            t[i-step] = (x1_mean - x2_mean) / (s * n)   
        return t

    #函数名：MKcount
    #参数：temp，reverse
    #注释：计算UFk,UBk 由于MK计算需要正反两遍计算，因此temp要单独传递，同时设定正序计算还是倒序计算
    def MKcount(self,temp,reverse= False):
        length = len(temp)
        Sk = np.zeros(length)       #创建一个Sk保存Sk的值
        Us = np.zeros(length)       #计算创建Us保存算出来的UFk
        s = 0
        for k in range(2,length):
            for i in range(1,k):
                if (temp[k]>temp[i]):
                    s += 1
            Sk[k]=s
            E=k*(k-1)/4
            var=k*(k-1)*(2*k+5)/72
            if (reverse == True):
                Us[k]=(E-Sk[k])/np.sqrt(var)
            else:
                Us[k]=(Sk[k] - E)/np.sqrt(var)
        return Us

    #函数名：MKplot
    #注释：根据获取到的UFk和UBk绘制图像并保存
    def MKplot(self,UFk,UBk):
        plt.clf()
        plt.title("Station {} Mann-Kendall mutation analysis".format(self.sta))
        l1 =plt.hlines(self.marked, self.year[0], self.year[-1], colors = "r", linestyles = "dashed") 
        plt.hlines(0, self.year[0], self.year[-1], colors = "y", linestyles = "-.")  #显著点
        plt.hlines(-self.marked, self.year[0], self.year[-1], colors = "r", linestyles = "dashed")
        plt.xticks(range(int(self.year[1]/5)*5,int(self.year[-1]/5)*5,5))
        l2,=plt.plot(self.year,UFk,marker=".")
        l3,=plt.plot(self.year,UBk,marker=".")
        plt.legend(handles=[l1,l2,l3],labels=['Marked=0.05',"UF","UB"],loc='upper left')
        plt.savefig("./台站{}MK检验.jpg".format(self.sta))

    #函数名：tplot
    #注释：该函数绘制出滑动t检验对应步长的图像并保存
    def tplot(self,t,step):
        # 初始化画布
        plt.clf()
        plt.figure(dpi=300,figsize=(8,4))
        plt.title("Sliding T-test with step of {}".format(step))
        l1 =plt.hlines(3.2, self.year[0], self.year[-1], colors = "r", linestyles = "dashed") #绘制显著点对应的线
        plt.hlines(0, self.year[0], self.year[-1], colors = "y", linestyles = "-.")  
        plt.hlines(-3.2, self.year[0], self.year[-1], colors = "r", linestyles = "dashed")
        plt.xticks(range(int(self.year[1]/5)*5,int(self.year[-1]/5)*5,5))
        # 绘制图像
        Xz = self.year[step:len(self.year)-step]
        l3, =plt.plot(Xz,t,marker=".")
        plt.legend(handles=[l1,l3],labels=['Marked=0.01','T'],loc='upper right')    #放置图例
        plt.xlabel("YEAR ")
        plt.ylabel("T")
        plt.savefig("./temp/{}+step={}.jpg".format(self.sta,step))
        # plt.show()

    #函数名：getYearsum
    #注释：从NOAA网站获取相应气象台站的每年数据总结，sta为台站代号,可以根据需要下载不同数据集
    def getYearsum(self): 
        if os.path.exists("./temp/"+self.sta+self.type+".csv"):
            print("检测到当前台站数据已存在，数据处理中")
            return True
        if (self.type == "day"):
            url = "https://www.ncei.noaa.gov/data/daily-summaries/access/"+self.sta+".csv"
        if (self.type == "month"):
            url = "https://www.ncei.noaa.gov/data/global-summary-of-the-month/access/"+self.sta+".csv"
        if (self.type == "year"):
            url = "https://www.ncei.noaa.gov/data/global-summary-of-the-year/access/"+self.sta+".csv"
        res = requests.get(url)
        if res.status_code == requests.codes.ok:
            with open("./temp/{}{}.csv".format(self.sta,self.type),"wb") as f:
                f.write(res.content)
            print("台站数据下载成功，正在进行数据处理")
            return True
        else:
            print("数据下载失败，请检查台站代码或网络链接是否正常,程序即将退出")
            return False

    #函数名：ReadData
    #注释：处理下载好的csv数据，读取相应年份和所需数据
    def ReadData(self):
        csv_dat = pd.read_csv("./temp/{}{}.csv".format(self.sta,self.type))
        data = csv_dat.dropna(subset=self.columns)                  #去除空数据所在的行
        year = np.array(pd.DataFrame(data,columns=["DATE"]))    #获取年份和平均温度
        temp = np.array(pd.DataFrame(data,columns=self.columns))
        year = year.reshape(1,len(year))[0]                     #将年份和平均温度数据处理
        temp = temp.reshape(1,len(temp))[0]
        if temp is None:
            print("请检查获取数据的列是否输入错误，程序退出")
            os._exit(1)
        return year,temp

if __name__ == "__main__":
    # sta = input("获取数据，请输入台站代码")
    #sta = "CHM00059287",step = [5,10,15],mark = 0.01, marked = 3.2,Dtype = "year",column = "TAVG"
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type = str, default="CHM00059287", help="台站在NOAA中的代码")
    parser.add_argument("--mark", type=int, default=0.01, help="置信度是多少")
    parser.add_argument("--marked", type=int, default=3.2, help="置信度对应的值是多少")
    parser.add_argument("--type", type=str, default="year", help="数据集的类型，有三个值，分别为year month day")
    parser.add_argument("--column", type=str, default="TAVG", help="需要取数据集的哪一列")
    parser.add_argument("--select",action="store_false",help="选择step是具体值还是通过range创建，默认使用range创建")
    parser.add_argument("--step", nargs="*", type=str, default="5,16,5", help="选择滑动的step值")
    opt = parser.parse_args()
    li = opt.step.split(",")
    li = list(map(int,li))
    if(opt.select):
        if(len(li) == 1):
            li = list(range(2,li[0]))
        if(len(li) == 2):
            li = list(range(li[0],li[1]))
        if(len(li) >= 3):
            li = list(range(li[0],li[1],li[2]))
    if li is None:
        print("请重新输入步长,程序退出")
        os._exit(1)
    process = DataProcess(opt.station, step = li, mark=opt.mark, marked=opt.marked, Dtype=opt.type, column=opt.column)
    process.Tprocess()
    process.MKprocess()
    #广州为CHM00059287

    

    
