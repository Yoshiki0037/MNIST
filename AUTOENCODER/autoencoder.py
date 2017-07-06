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


class AutoEncoder(chainer.Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__(
                l1=L.Linear(784, 100),
                l2=L.Linear(100, 784)
                )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y

if __name__ == '__main__':
    gpu = 0
    num_epochs = 10
    batch_size = 500
    learning_rate = 0.001

    xp = cuda.cupy if gpu >= 0 else np

    train, test = load_mnist(1)
    x_train, c_train = train
    x_test, c_test = test
    num_train = len(x_train)
    num_test = len(x_test)

    model = AutoEncoder()
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
        plt.matshow(x.reshape(28, 28), cmap=plt.cm.gray)
        plt.show()
        plt.matshow(y_batch[i].reshape(28, 28), cmap=plt.cm.gray)
        plt.show()





















