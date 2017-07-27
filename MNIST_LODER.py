# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:53:19 2017

@author: kawalab
"""

import urllib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gzip


# make cashe for images
def make_images_np(file_path):
    
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
    num_images = int.from_bytes(data[4:8], 'big')
    width = int.from_bytes(data[8:12], 'big')
    height = int.from_bytes(data[12:16], 'big')
    pixels = np.frombuffer(data, np.uint8, -1, 16)
    images = pixels.reshape(num_images, width, height, 1)
    
    return images


# make cashe for labels
def make_labeles_np(file_path):
    
    with gzip.open(file_path, 'rb') as f:
        data = f.read()
    labels = np.frombuffer(data, np.uint8, -1, 8)
    
    return labels
    

# データが残っていたら利用, 残っていなければキャッシュ作成
def mnist_loader(ndim=2, dataset_root='c:/dataset'):
    dataset_dir = Path(dataset_root) / 'mnist'
    if not dataset_dir.exists():
        dataset_dir.mkdir(exist_ok=True)
        
    # Data_file_path
    train_images_file = 'train-images-idx3-ubyte.gz'
    test_images_file = 't10k-images-idx3-ubyte.gz'
    train_labels_file = 'train-labels-idx1-ubyte.gz'
    test_labels_file = 't10k-labels-idx1-ubyte.gz'
    
    # cache_file_path
    train_image_cache = dataset_dir / 'train_images.npy'
    test_image_cache = dataset_dir / 'test_images.npy'
    train_labels_cache = dataset_dir / 'train_labels.npy'
    test_labels_cache = dataset_dir / 'test_labels.npy'

    root_url = 'http://yann.lecun.com/exdb/mnist/' 
    
    train_image_path = dataset_dir / train_images_file
    if not Path(train_image_cache).exists():
        urllib.request.urlretrieve(root_url + train_images_file,
                                   train_image_path)  # データファイルのDL
                                   
        train_images_data = make_images_np(train_image_path)
        np.save(train_image_cache, train_images_data) # ndarrayをファイルに保存する
        print('makeing cache for train images is completed.')
    else:
        train_images_data = np.load(train_image_cache)
        
    test_image_path = dataset_dir / test_images_file
    if not Path(test_image_cache).exists():
        urllib.request.urlretrieve(root_url + test_images_file, 
                                  test_image_path) #データファイルのダウンロード
        
        test_images_data = make_images_np(test_image_path)
        np.save(test_image_cache, test_images_data)
        print('making cache for test images is completed')
    else:
        test_images_data = np.load(test_image_cache)
        
    train_labels_path = dataset_dir / train_labels_file
    if not Path(train_labels_cache).exists():
        urllib.request.urlretrieve(root_url + train_labels_file, 
                                  train_labels_path) #データファイルのダウンロード
        
        train_labels_data = make_labeles_np(train_labels_path)
        np.save(train_labels_cache, train_labels_data)
        print('making cache for tairn labels is completed')
    else:
        train_labels_data = np.load(train_labels_cache)
    
    test_labels_path = dataset_dir / test_labels_file
    if not Path(test_labels_cache).exists():
        urllib.request.urlretrieve(root_url + test_labels_file, 
                                  test_labels_path) #データファイルのダウンロード
        
        test_labels_data = make_labeles_np(test_labels_path)
        np.save(test_labels_cache, test_labels_data)
        print('making cache for tairn labels is completed')
    else:
        test_labels_data = np.load(test_labels_cache)
        
    if ndim == 1:
        train_images_data = train_images_data.reshape(-1, 28 * 28)
        test_images_data = test_images_data.reshape(-1, 28 * 28)
    
    elif ndim == 2:
        train_images_data = train_images_data.reshape(-1, 28, 28)
        test_images_data = test_images_data.reshape(-1, 28, 28)
        
    elif ndim == 3:
        train_images_data = train_images_data.reshape(-1, 1, 28 * 28)
        test_images_data = test_images_data.reshape(-1, 1, 28 * 28)
    
    else:
        raise ValueError('you need define ndim between from 1 to 3')
        
    return (train_images_data, test_images_data, 
            train_labels_data, test_labels_data)
    

if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels = mnist_loader(2)
    a = train_images[0]
    b = test_images[0]
    plt.matshow(a, cmap=plt.cm.gray)    
    plt.show()
    plt.matshow(b, cmap=plt.cm.gray)
    plt.show()
    print('train label = {}'. format(train_labels[0]))
    print('test label = {}'. format(test_labels[0]))
    