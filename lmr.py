# -*- coding: utf-8 -*-

#0.mpg (continuous)
#1.cylinders
#2.displacement (continuous)
#3.horsepower (continuous)
#4.weight (continuous)
#5.acceleration (continuous)
#6.model year
#7.origin
#8.car name

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

datapre=pd.read_csv("auto-mpg.csv", header=None, sep='\s+') #ファイル読み込み
datapre=datapre[datapre[3] != '?']
data = np.array(datapre)

N = data.shape[0] #データ数

def lmr(A,B,order): #A,Bは特徴量、orderは表示順
    t=data[:,0] #目標値
    x1 = data[:,A].astype(np.float32)
    x2 = data[:,B].astype(np.float32)
    x3 = np.ones(N)
    X=np.c_[x1,x2,x3] #特徴ベクトル
 
    w = np.dot(np.linalg.pinv(X),t) #線形重回帰
    print w
    
    xad=(max(x1)-min(x1))/20 #平面生成
    xbd=(max(x2)-min(x2))/20
    xa = np.arange(min(x1)-xad, max(x1)+xad, xad) 
    xb = np.arange(min(x2)-xbd, max(x2)+xbd, xbd)
    Xa, Xb = np.meshgrid(xa, xb)
    Y = w[0]*Xa + w[1]*Xb + w[2]

    ax = fig.add_subplot(1,2,order, projection='3d')
    ax.plot_wireframe(Xa, Xb, Y) #平面プロット
    for i3 in range(N): #散布図
        ax.scatter(x1[i3],x2[i3],t[i3], c='b', s=20)

    ax.set_title("graph {0} and {1}" .format(A,B))
    ax.set_xlabel("data {0}" .format(A))
    ax.set_ylabel("data {0}" .format(B))
    ax.set_zlabel("mpg")
    

fig = plt.figure()
lmr(4,3,1)
lmr(2,5,2)
plt.show()

