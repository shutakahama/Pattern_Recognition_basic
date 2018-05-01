#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *

#トレーニングデータ読み込み
#画素値は各画像の28*28=784画素を一行に表示。濃淡の256段階を、0〜1の範囲で表現するようにした。
#ラベルについて、数字iのラベルはi番目が1で他は0の10次元ベクトルで表現
X_train = np.loadtxt('train-images5000.txt').astype(np.float32) / 255.0
y_train = np.loadtxt('train-labels5000.txt').astype(np.int32)
y_train = np.eye(10)[y_train].astype(np.int32)
N_train = len(X_train)

# テストデータ
X_test = np.loadtxt('test-images1000.txt').astype(np.float32) / 255.0
y_test = np.loadtxt('test-labels1000.txt').astype(np.int32)
y_test = np.eye(10)[y_test].astype(np.int32)
N_test = len(X_test)
'''
#テストデータの正規化
X_train_mean = np.mean(X_train, axis=1, keepdims=True)
X_train_var = np.var(X_train, axis=1, keepdims=True)
X_train = (X_train - X_train_mean)/X_train_var
X_test_mean = np.mean(X_test, axis=1, keepdims=True)
X_test_var = np.var(X_test, axis=1, keepdims=True)
X_test = (X_test - X_test_mean)/X_test_var
'''
#中間層での活性化関数はSigmoid
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def deriv(self, x):#微分
        return self(x) * (1 -  self(x))

#出力層での活性化関数はSoftmax
class Softmax:
    def __call__(self, x):
        c = np.max(x, axis=1)[:, np.newaxis]
        return np.exp(x - c) / np.sum(np.exp(x - c), axis=1, keepdims=True)

    def deriv(self, x):#微分
        return self(x) * (1 -  self(x))

#各層のユニット内での処理
class Linear:
    def __init__(self, in_dim, out_dim, activation):#パラメータ等の初期化
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim).astype(np.float32)
        self.delta = None
        self.activation = activation()

    def __call__(self, x):#y = f(Wx + b)
        self.u = np.dot(x, self.W) + self.b
        self.z = self.activation(self.u)
        return self.z

#層と伝搬計算の定義
class MLP():
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x, t, lr):     
        # 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        
        # 誤差逆伝播
        # 最終層の誤差
        delta = self.y - t #dL/du = y - t
        self.layers[-1].delta = delta
        W = self.layers[-1].W
        
        # 中間層の誤差を計算
        # 各ループ開始時に、一つ上の層の誤差と重みがそれぞれdelta、Wに格納されている
        for layer in self.layers[-2::-1]:
            delta = np.dot(delta, W.T) * layer.activation.deriv(layer.u) #dL/du = (dL(-1)/du*W)*f'(u)
            layer.delta = delta
            W = layer.W
        
        # 各層のパラメータを更新
        # 各ループ開始時に、一つ下の層の出力がzに格納されている
        z = x
        for layer in self.layers:
            
            dW = np.dot(z.T, layer.delta)
            db = np.dot(np.ones(len(z)), layer.delta)
            layer.W -= lr * dW #W = W - lr(dL/dW)
            layer.b -= lr * db #b = b - lr(dL/db)
            z = layer.z
            
        return self.loss
        
    def test(self, x, t):
        # 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss


#具体的な層の定義
#入力層は784(28*28)入力1000出力、中間層は1000入力1000出力、出力層は1000入力10(0〜9の判別)出力
model = MLP([Linear(784, 1000, Sigmoid),
             Linear(1000, 1000, Sigmoid),
             Linear(1000, 10, Softmax)])

n_epoch = 15 #学習の繰り返し回数
batchsize = 100 #ミニバッチの個数
lr = 0.01 #学習係数
losses = {'train':[], 'test': []}
accuracies = {'train':[], 'test':[]}

#学習の開始
for epoch in range(n_epoch):
    print('epoch %d ' % (epoch+1))  

    # トレーニング
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(N_train)

    #ミニバッチ生成、誤差伝搬を行い、学習誤差を計算する
    for i in range(0, N_train, batchsize):
        x = X_train[perm[i:i+batchsize]]
        t = y_train[perm[i:i+batchsize]]
        
        sum_loss += model.train(x, t, lr) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))
    
    loss = sum_loss / N_train #誤差関数の値の平均
    losses['train'].append(loss)
    accuracy = np.sum(np.eye(10)[pred_y] * y_train[perm]) / N_train #出力で最大となった成分と正解ラベルの内積を取ることで正答率を計算
    accuracies['train'].append(accuracy)
    print('Train loss %.3f, accuracy %.4f ' %(loss, accuracy))
    
    
    # テスト
    sum_loss = 0
    pred_y = []
    
    for i in range(0, N_test, batchsize):
        x = X_test[i: i+batchsize]
        t = y_test[i: i+batchsize]

        sum_loss += model.test(x, t) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))
            
    loss = sum_loss / N_test
    losses['test'].append(loss)
    accuracy = np.sum(np.eye(10)[pred_y] * y_test) / N_test
    accuracies['test'].append(accuracy)
    print('Test  loss %.3f, accuracy %.4f' %(loss, accuracy))


#可視化
#誤差関数
plt.figure()
plt.subplot(1,2,1)
plt.plot(np.arange(1, n_epoch + 1), losses['train'], 'blue', label='Train Loss')
plt.plot(np.arange(1, n_epoch + 1), losses['test'], 'orange', label='Test Loss')

plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.title("Losses")

#正答率（トレーニングデータ、テストデータ）
plt.subplot(1,2,2)
plt.plot(np.arange(1, n_epoch + 1), accuracies['train'], \
         'blue', label='Train Accuracy')
plt.plot(np.arange(1, n_epoch + 1), accuracies['test'], \
         'orange', label='Test Accuracy')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracies")
plt.legend()
plt.show()

