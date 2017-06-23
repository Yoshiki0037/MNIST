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


def load_mnist():
    train, test = get_mnist(ndim=3)
    train = concat_examples(train)
    test = concat_examples(test)
    return train, test


class ConvNet(chainer.Chain):
    def __init__(self):
        super(ConvNet, self).__init__(
            conv1=L.Convolution2D(1, 100, 3),
            conv2=L.Convolution2D(100, 100, 4),
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
    gpu = 0
    num_epochs = 10
    batch_size = 500
    learning_rate = 0.001

    xp = cuda.cupy if gpu >= 0 else np

    train, test = load_mnist()
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    model = ConvNet()
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    train_loss_log = []
    train_acc_log = []
    test_loss_log = []
    test_acc_log = []
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        epoch_losses = []               # エポック内の損失値
        epoch_accs = []                 # エポック内の認識率
        for i in tqdm(range(0, num_train, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            c_batch = xp.asarray(c_train[i:i+batch_size])
            y_batch = model(x_batch)

            # 損失関数の計算
            loss = F.softmax_cross_entropy(y_batch, c_batch)
            model.cleargrads()              # 勾配のリセット
            loss.backward()                 # 重みの更新
            accuracy = F.accuracy(y_batch, c_batch)       # 認識率
            optimizer.update()

            epoch_losses.append(loss.data)
            epoch_accs.append(accuracy.data)

        epoch_loss = np.mean(cuda.to_cpu(xp.stack(epoch_losses)))   # エポックの平均損失
        epoch_acc = np.mean(cuda.to_cpu(xp.stack(epoch_accs)))     # エポックの平均認識率
        train_loss_log.append(epoch_loss)
        train_acc_log.append(epoch_acc)

        losses = []
        accs = []
        for i in tqdm(range(0, num_test, batch_size)):
            x_batch = xp.asarray(x_train[i:i+batch_size])  # 1->バッチサイズまでのループ
            c_batch = xp.asarray(c_train[i:i+batch_size])

            x_batch = chainer.Variable(x_batch, volatile=True)
            c_batch = chainer.Variable(c_batch, volatile=True)
            y_batch = model(x_batch)

            loss = F.softmax_cross_entropy(y_batch, c_batch)
            accuracy = F.accuracy(y_batch, c_batch)

            losses.append(loss.data)
            accs.append(accuracy.data)
        test_loss = np.mean(cuda.to_cpu(xp.stack(losses)))   # エポックの平均損失
        test_acc = np.mean(cuda.to_cpu(xp.stack(accs)))     # エポックの平均認識率
        test_loss_log.append(test_loss)
        test_acc_log.append(test_acc)

        if loss.data < best_val_loss:
            best_model = deepcopy(model)
            best_val_loss = loss.data
            best_epoch = epoch

        print('{}: acc={}, loss={}'.format(epoch, epoch_acc, epoch_loss))

        plt.figure(figsize=(10, 4))
        plt.title('Loss')
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_log, label='train_loss')
        plt.plot(test_loss_log, label='train_acc')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.plot(train_acc_log)
        plt.plot(test_acc_log)
        plt.ylim([0.0, 1.0])
        plt.legend(['val loss', 'val acc'])
        plt.grid()

        plt.tight_layout()
        plt.show()
