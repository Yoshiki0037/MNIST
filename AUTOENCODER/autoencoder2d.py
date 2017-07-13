# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:02:21 2017

@author: kawalaboon
"""

from copy import deepcopy   # 深い複製

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer.datasets import get_mnist
from chainer.dataset import concat_examples


def load_mnist(ndim):
    train, test = get_mnist(ndim=ndim)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


class AutoEncoder2D(chainer.Chain):
    def __init__(self):
        super(AutoEncoder2D, self).__init__(
                conv1=L.Convolution2D(1, 32, 1, pad=1),
                conv2=L.Convolution2D(32, 64, 3),
                conv3=L.Convolution2D(64, 128, 3),
                conv4=L.Convolution2D(128, 384, 3),
                conv5=L.Convolution2D(384, 1152, 3),

                dconv5=L.Deconvolution2D(1152, 384, 3),
                dconv4=L.Deconvolution2D(384, 128, 3),
                dconv3=L.Deconvolution2D(128, 64, 3),
                dconv2=L.Deconvolution2D(64, 32, 3),
                dconv1=L.Deconvolution2D(32, 1, 1, pad=1)
                )

    def __call__(self, x):
        outsize1=x.shape[-2:]        
        h = F.relu(self.conv1(x))                   #26
        h = F.relu(self.conv2(h))                   #24
        h = F.max_pooling_2d(h, 2)                  #12
        h = F.relu(self.conv3(h))                   #10
        h = F.relu(self.conv4(h))                   #8      
        outsize2=h.shape[-2:]        
        h = F.max_pooling_2d(h, 2)                  #4
        h = F.relu(self.conv5(h))                   #2
        
                  
        h = F.relu(self.dconv5(h))                 #4
        h = F.unpooling_2d(h, 2, outsize=outsize2)  #8
        h = F.relu(self.dconv4(h))                 #10
        h = F.relu(self.dconv3(h))                 #12
        h = F.unpooling_2d(h, 2, outsize=outsize1)  #24
        h = F.relu(self.dconv2(h))                 #26
        y = F.sigmoid(self.dconv1(h))               # 28
        return y
        

if __name__ == '__main__':
    gpu = 0
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.0001

    xp = cuda.cupy if gpu >= 0 else np

    train, test = load_mnist(3)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    model = AutoEncoder2D()
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    train_loss_log = []
    test_loss_log = []
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        epoch_losses = []               # エポック内の損失値
        for i in tqdm(range(0, num_train, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.mean_squared_error(y_batch, x_batch)
            model.cleargrads()              # 勾配のリセット
            loss.backward()                 # 重みの更新
            optimizer.update()

            epoch_losses.append(loss.data)
            
        epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))   # エポックの平均損失
        train_loss_log.append(epoch_loss)

        losses = []
        for i in tqdm(range(0, num_test, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ

            x_batch = chainer.Variable(x_batch, volatile=True)
            y_batch = model(x_batch)

            loss = F.mean_squared_error(y_batch, x_batch)

            losses.append(loss.data)
        test_loss = np.mean(cuda.to_cpu(xp.stack(losses)))   # エポックの平均損失
        test_loss_log.append(test_loss)

        if loss.data < best_val_loss:
            best_model = deepcopy(model)
            best_val_loss = loss.data
            best_epoch = epoch

        print('{}: loss={}'.format(epoch, epoch_loss))

        plt.figure(figsize=(10, 4))
        plt.title('Loss')
        plt.plot(train_loss_log, label='train_loss')
        plt.plot(test_loss_log, label='train_acc')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
    n=15
    x_batch = xp.asarray(x_test[:n])
    y_batch = best_model(x_batch)   
    y_batch = cuda.to_cpu(y_batch.data)
    for i in range(n):
        x=x_test[i]
        plt.matshow(cuda.to_cpu(x_batch[i][0]), cmap=plt.cm.gray)
        plt.show()
        plt.matshow(y_batch[i][0], cmap=plt.cm.gray)
        plt.show()





















