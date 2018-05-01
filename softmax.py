#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#トレーニングデータ読み込み
#画素値は各画像の28*28=784画素を一行に表示。濃淡の256段階を、0〜1の範囲で表現するようにした。
#ラベルについて、数字iのラベルはi番目が1で他は0の10次元ベクトルで表現
X_train = np.loadtxt('train-images5000.txt').astype(np.float32) / 255.0
y_train = np.loadtxt('train-labels5000.txt').astype(np.int32)
y_train = np.eye(10)[y_train].astype(np.int32)
N_train = len(X_train)

# テストデータ読み込み
X_test = np.loadtxt('test-images1000.txt').astype(np.float32) / 255.0
y_test = np.loadtxt('test-labels1000.txt').astype(np.int32)
y_test = np.eye(10)[y_test].astype(np.int32)
N_test = len(X_test)

#活性化関数Softmax
def Softmax(x):
    c = np.max(x, axis=1)[:, np.newaxis]
    return np.exp(x - c) / np.sum(np.exp(x - c), axis=1, keepdims=True)

#順伝播計算
def fp(x, W, b):#y = f(Wx + b)
    u = np.dot(x, W) + b
    return Softmax(u)



times = 10 #学習の繰り返し回数
batchsize = 100 #ミニバッチの個数
lr = 0.01 #学習係数
losses_train = []
losses_test = []
accuracies_train = []
accuracies_test = []

#重み等の初期化
W = np.random.uniform(low=-0.08, high=0.08, size=(784, 10)).astype(np.float32)
b = np.zeros(10).astype(np.float32)
delta = None



for time in range(times):
    print('times %d ' % time)
    
    # トレーニング
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(N_train) #バッチ生成のため、trainデータを並び替え

    #ミニバッチ生成、誤差伝搬を行い、学習誤差を計算する
    for i in range(0, N_train, batchsize):
        x = X_train[perm[i:i+batchsize]]
        t = y_train[perm[i:i+batchsize]]

        #順伝播
        y = fp(x, W, b)
        loss = np.sum(-t*np.log(y + 1e-7)) / len(x)
        sum_loss += loss 

        #逆伝播、更新
        delta = y - t #dL/du = p - t
        dW = np.dot(x.T, delta) 
        db = np.dot(np.ones(len(x)), delta)
        W -= lr * dW #W = W - lr(dL/dW)
        b -= lr * db #b = b - lr(dL/db)

        pred_y.extend(np.argmax(y, axis=1))
    
    loss = sum_loss / N_train #誤差関数の値の平均
    accuracy = np.sum(np.eye(10)[pred_y] * y_train[perm]) / N_train #出力で最大となった成分と正解ラベルの内積を取ることで正答率を計算
    losses_train.append(loss)
    accuracies_train.append(accuracy)
    print('Train loss %.3f, accuracy %.4f ' %(loss, accuracy))
    
    
    # テスト
    sum_loss = 0
    pred_y = []
    
    for i in range(0, N_test, batchsize):
        x = X_test[i: i+batchsize]
        t = y_test[i: i+batchsize]
        
        y = fp(x, W, b)
        loss = np.sum(-t*np.log(y + 1e-7)) / len(x)
        sum_loss += loss
        pred_y.extend(np.argmax(y, axis=1))

    loss = sum_loss / N_test
    accuracy = np.sum(np.eye(10)[pred_y] * y_test) / N_test
    losses_test.append(loss)
    accuracies_test.append(accuracy)
    print('Test  loss %.3f, accuracy %.4f' %(loss, accuracy))



#可視化
#誤差関数
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.arange(1, times + 1), losses_train, 'blue', label='Train Loss')
plt.plot(np.arange(1, times + 1), losses_test, 'orange', label='Test Loss')
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.title("Losses")

#正答率（トレーニングデータ、テストデータ）
plt.subplot(1,2,2)
plt.plot(np.arange(1, times + 1), accuracies_train, 'blue', label='Train Accuracy')
plt.plot(np.arange(1, times + 1), accuracies_test, 'orange', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracies")
plt.legend()
plt.show()

