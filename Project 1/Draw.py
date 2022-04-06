import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
## Task 1

# result1=[0.5, 0.397, 0.317, 0.201, 0.199, 0.209]
# result2=[0.517, 0.461, 0.443,0.782]
# result1=[1-i for i in result1]
# result2=[1-i for i in result2]
# x1=['10','20','50','100','200','500']
# x2=['8','8,8','8,8,8','8,8,8,8']
# p1 = plt.bar(x1, result1,width=0.45,label="单隐藏层")
# p2=  plt.bar(x2, result2,width=0.45,label="多隐藏层")

# plt.title('不同设置隐藏层准确率', fontproperties="SimSun")
# plt.xlabel('nHidden')
# plt.legend()
# plt.show()


## Task 2

# x1=[1,3,5]
# y=[0.250,0.530,0.403]
# n=[0.233,0.621,0.543]
# y=[1-i for i in y]
# n=[1-i for i in n]
# unit=0.6
# ysr = ['64','64,8','10,10']   #x标签
# y0  = ['' for i in range(len(ysr))]       #空x标签
# x2=[i+unit*1.1 for i in x1]   #bias
# group_center=[i+unit/2*1.1 for i in x1]  #组中心
# center_y=[0 for i in range(len(x1))]
# plt.bar(x1, n, alpha=0.7, width=unit, color='r',label="raw", tick_label=y0)
# plt.bar(x2, y, alpha=0.7, width=unit, color='g',label="momentum", tick_label=y0)
# plt.bar(group_center,center_y,tick_label=ysr)
# plt.title('带惯性与不带惯性准确率对比柱状图', fontproperties="SimSun")
# plt.xlabel('nHidden')
# plt.legend() # 显示图例
# plt.show()



## Task 5

# y1=[1-0.068,1-0.089]
# x=[2,4]
# tN=['softmax','squared']
# plt.bar(x,y1,alpha=0.7,width=0.6)
# plt.title('20万次迭代下不同损失函数模型精度对比')
# plt.xlim(1,5)
# for i in range(2):
#     plt.text(x[i]+0.05,y1[i]+0.01,'%.2f'%y1[i], ha='center',va='bottom')
# plt.xticks(x,tN)
# # plt.legend()
# plt.show()


## Task 8

# y1=[0.202000,0.232000] #lr=1
# y2=[0.164000,0.168000] #lr=0.5
# y2=[0.195,0.193]   #lr=0.1
# y3=[0.169,0.168]   #lr=0.05

# y1=[0.202,0.164,0.195,0.169]
# y2=[0.232,0.168,0.193,0.168]
# y1=[1-i for i in y1]
# y2=[1-i for i in y2]


# unit=0.6
# ysr = ['lr=1','lr=0.5','lr=0.1','lr=0.05']   #x标签
# y0  = ['' for i in range(len(ysr))]       #空x标签
# x1=[1,5,9,13]
# x2=[i+unit*1.1 for i in x1]   #bias
# group_center=[i+unit/2*1.1 for i in x1]  #组中心
# center_y=[0 for i in range(len(x1))]
# plt.bar(x1, y1, alpha=0.7, width=unit, color='r',label="不微调", tick_label=y0)
# plt.bar(x2, y2, alpha=0.7, width=unit, color='g',label="微调", tick_label=y0)
# plt.bar(group_center,center_y,tick_label=ysr)
# plt.title('不同微调率模型与不微调模型准确度对比', fontproperties="SimSun")
# plt.xlabel('learning rate')
# plt.legend() # 显示图例
# plt.ylim(0.75,0.88)
# plt.show()


## Task 9

# y1=[1-0.169,1-0.170]
# x=[2,4]
# tN=['增加样本','不增加样本']
# plt.bar(x,y1,alpha=0.7,width=0.6)
# plt.title('人工增加样本与否模型精度对比')
# plt.xlim(1,5)
# plt.ylim(0.8,0.86)
# for i in range(2):
#     plt.text(x[i]+0.05,y1[i],'%.3f'%y1[i], ha='center',va='bottom')
# plt.xticks(x,tN)
# # plt.legend()
# plt.show()


## Task 10

# y1=[1-0.180,1- 0.378]
# x=[2,4]
# tN=['卷积层','全连接层']
# plt.bar(x,y1,alpha=0.7,width=0.6)
# plt.title('第一层为卷积层或全连接层模型精度对比')
# plt.xlim(1,5)
# plt.ylim(0.5,0.85)
# plt.xticks(x,tN)
# plt.xlabel('第一层类型')
# # plt.legend()
# plt.show()


## Task 3

y1=[22.403,62.165]
x=[2,4]
tN=['向量化','非向量化']
plt.bar(x,y1,alpha=0.7,width=0.6)
plt.title('向量化与否运行时间对比')
plt.xlim(1,5)
plt.xticks(x,tN)

plt.ylabel('秒(s)')
# plt.legend()
plt.show()