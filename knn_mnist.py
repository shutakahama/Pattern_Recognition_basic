# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

# トレーニングデータ読み込み
X_train = np.loadtxt('./data/train-images5000.txt').astype(np.float32) / 255.0
y_train = np.loadtxt('./data/train-labels5000.txt').astype(np.int32)
y_train = np.eye(10)[y_train].astype(np.int32)
N_train = len(X_train)


# テストデータ読み込み
X_test = np.loadtxt('./data/test-images1000.txt').astype(np.float32) / 255.0
y_test = np.loadtxt('./data/test-labels1000.txt').astype(np.int32)
y_test = np.eye(10)[y_test].astype(np.int32)
N_test = len(X_test)

print('train num: {}, test num: {}'.format(N_train, N_test))

# データの正規化
X_train_mean = np.mean(X_train, axis=1, keepdims=True)
X_train_var = np.var(X_train, axis=1, keepdims=True)
X_train = (X_train - X_train_mean)/X_train_var
X_test_mean = np.mean(X_test, axis=1, keepdims=True)
X_test_var = np.var(X_test, axis=1, keepdims=True)
X_test = (X_test - X_test_mean)/X_test_var

# k近傍法のkの値
K = 20
accuracy = 0

for i in tqdm(range(N_test)):
    length = [[0 for col in range(2)] for row in range(N_train)]
    for j in range(N_train):
        length[j][0] = j
        # 各train dataとの距離を測る
        length[j][1] = np.dot((X_test[i] - X_train[j]), (X_test[i] - X_train[j]))
                
    # 距離の順に並べ替え
    length.sort(key = lambda x:x[1])
    
    pre_y = np.zeros(10)
    # pre_yという10次元ベクトルの各成分が近傍にある0〜9の個数になる
    for n in range(2, K+2):
        pre_y += y_train[length[n][0]]

    # pre_yのうち最大の成分が予測する答え
    ans_y = np.eye(10)[np.argmax(pre_y)]
    # ラベルと一致する場合のみ1が足される
    accuracy += np.dot(ans_y, y_test[i])
    # print('test {} : label {}, reliability {}'.format(i+1, np.argmax(pre_y), np.max(pre_y)/K))
    # print('accurate count : {}'.format(accuracy))

accuracy /= N_test
print(accuracy)
