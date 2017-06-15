# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:02:21 2017

@author: kawalaboon
"""

import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples


def load_mnist_as_ndarray():
    train, test = get_mnist(ndim=3)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


class ConvNet(chainer.Chain):
    def __init__(self):
        super(ConvNet, self).__init__(
            conv1=L.Convolution2D(1, 100, 3),  # 28 -> 26
            conv2=L.Convolution2D(100, 100, 4),  # 13 -> 10
            fc1=L.Linear(2500, 10)
        )
        
     
    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, 2)
        h = self.conv2(h)
        h = F.max_pooling_2d(h, 2)
        y = self.fc1(h)
        return y
    
if __name__ == '__main__':
    num_epochs = 100
    batch_size = 500
    learning_rate = 0.001       
        
    # データ読み込み
    train, test = load_mnist_as_ndarray()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)
        
    # モデルとオプティマイザの準備
    model = ConvNet()
    optimaizer = optimizers.Adam(learning_rate)
    optimaizer.setup(model)        
        
    # 訓練ループ
    loss_log = []
    for epoch in range(num_epochs):
       for i in range(0, num_train, batch_size):
         x_batch = x_train[i:i+batch_size]
         c_batch = c_train[i:i+batch_size]
         y_batch = model(x_batch)
                
         loss = F.softmax_cross_entropy(y_batch, c_batch)
         model.cleargrads()
         loss.backward()
         optimaizer.update()
                             
         accuracy = F.accuracy(y_batch, c_batch)
         print(epoch, accuracy.data, loss.data)
         loss_log.append(loss.data)
                
plt.plot(loss_log)
plt.show()  
        