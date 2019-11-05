# Pettern_Recognition_basic

このリポジトリには，講義の課題として作成したパターン認識に関する基本的な手法が実装されています．  
dataフォルダには分析に使うデータファイルがあります．  
- auto-mpg  
- fisher  
- MNIST

resultフォルダには結果のグラフがあります．  

## 概要
### knn.py  
k近傍法による分類の実装．kの値を変化させた時の正解率をグラフで出力．
### knn_mnist.py  
k近傍法によるMNIST分類の実装．デフォルトでk=20．
### kmean.py  
k-meanによる教師なしクラスタリングの実装．デフォルトではk=2,3,4,5で変化させた時の正解率を出力．
### lmr.py  
線形重回帰(Linear Multiple Regression)の実装．3次元で回帰した時の平面をグラフで出力．
### softmax.py  
ソフトマックス回帰の実装．デフォルトでは10epoch回して正解率を出力．
### neuralnet.py  
numpyによるニューラルネットの実装．デフォルトでは10epoch回して正解率を出力．
