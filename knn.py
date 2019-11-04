# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("./data/fishers.txt", dtype = None, delimiter = "\t")  # ファイルから読み込み
print(data)
number = 149
percent = [0 for i in range(30)]

for k in range(1, 31):
    counter = 0
    for i in range(number):
        length = [[0 for col in range(2)] for row in range(149)]
        for j in range(number):  # 各点との距離をはかり、データの順番とともに配列に入れる
            length1 = data[i][0] - data[j][0]
            length2 = data[i][1] - data[j][1]
            length3 = data[i][2] - data[j][2]
            length4 = data[i][3] - data[j][3]
            length[j][0] = j
            length[j][1] = np.sqrt(length1**2 + length2**2 + length3**2 + length4**2)
            
        length.sort(key = lambda x:x[1])  # 距離の順に並べ替え
        seto = ver = vir = 0
        for n in range(2, k+2):  # 近くにあるk個のデータについて、ラベルの数を数える。最も近いのは自分自身なので除外
            p = length[n][0]
            if data[p][4] == b'I. setosa':
                seto += 1
            elif data[p][4] == b'I. versicolor':
                ver += 1
            elif data[p][4] == b'I. virginica':
                vir += 1
                
        if seto >= ver and seto >= vir:  # 最大個数のものとi番目のラベルが一致すればcounterを増やす
            if data[i][4] == b'I. setosa':
                counter += 1
        elif ver > seto and ver >= vir:
            if data[i][4] == b'I. versicolor':
                counter += 1
        elif vir > seto and vir > ver:
            if data[i][4] == b'I. virginica':
                counter += 1

    percent[k-1] = 100*float(counter)/(number-1)  # 各iについてcounterを足し、正答率を出す

print(percent)

# グラフに表示
xaxis = [i for i in range(1,31)] 
plt.plot(xaxis, percent, color = "red")
plt.title("k-NN (k = 1~30)")
plt.xlabel("k")
plt.ylabel("rate of correct answer(%)")
plt.savefig('./result/knn_result.png')
# plt.show()
