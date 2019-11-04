# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy

data = np.genfromtxt("./data/fishers.txt", dtype = None, delimiter = "\t")  # ファイルから読み込み
number = 149  # サンプル数

for knum in range(2,6):  # knum:クラスタの個数
    loop = 0
    cls = [0 for i1 in range(number)]

    # クラスタの中心の初期値
    center = [[3, 1, 3, 1.5],
              [5, 5, 3, 1.5],
              [10, 5, 5, 2],
              [8, 2, 2, 1.5],
              [7, 4, 1, 2]]
    
    centerold = [[1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1]]

    # 各中心が移動しなくなるまで
    while centerold != center and loop < 30:
        centerpre = [[0 for i2 in range(4)] for i3 in range(knum)]
        count = [0 for i4 in range(knum)]
        length = [0 for i5 in range(knum)]
        lengthpre = [[0 for i6 in range(4)] for i7 in range(knum)]

        centerold = copy.deepcopy(center)

        # 各中心との距離を求める
        for i in range(number):
            for g in range(knum):
                for j in range(4):
                    lengthpre[g][j] = data[i][j] - center[g][j]
                length[g] = np.sqrt(lengthpre[g][0]**2 + lengthpre[g][1]**2 +
                                    lengthpre[g][2]**2 + lengthpre[g][3]**2)

            # 各中心の中で最小距離のものを調べ、クラスに分類する
            for g in range(knum):
                if min(length) == length[g]:
                    cls[i] = g+1
                    count[g] += 1
                    for j in range(4):
                        centerpre[g][j] += data[i][j]

        # クラスごとに足し合わされた値を割って中心座標を求める
        for g in range(knum):
            if count[g] == 0:
                print("!!invalid clustering")
            else:
                for i in range(4):
                    center[g][i] = centerpre[g][i]/count[g]

        loop += 1

    print("k: {}, loop: {}".format(knum, loop))

    # 分類後のデータのプロット
    color =["red", "blue", "purple", "green", "yellow"]
    plt.subplot(2, 2, knum-1)
    plt.title('kmean {0}'.format(knum))
    for i in range (number):
        for g in range(knum):
            if cls[i] == g+1:
                plt.scatter(data[i][0], data[i][1], c=color[g])

    # 中心のプロット
    for g in range(knum):
        plt.scatter(center[g][0], center[g][1], s=300, c="black", marker="*")

plt.savefig("./result/kmean.png")
# plt.show()
