# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random


def get_one_hot(labels, num_classes):
    """one_hot编码
    Args:
        labels: 输入类别标签
        num_classes: 类别个数
    """
    one_hot = np.zeros([labels.shape[0], num_classes])
    for i in range(labels.shape[0]):
        one_hot[i][labels[i]] = 1
    return one_hot


def generate(sample_szie, mean, cov, diff, num_classes=2, one_hot=False):
    """按照指定的均值和方差生成固定数量的样本

    Args:
        sample_szie: 样本个数
        mean: 长度为M的一维ndarray或list 对应每个特征的均值
        cov: N x N 的ndarray或list 协方差 对称矩阵
        diff: 长度为 类别-1 的list  每i元素为第i个类别和第0个类别均值的差值 [特征1差，特征2差....]  如果长度不够，后面每个元素值取diff最后一个元素
        num_classes: 分类数
        one_hot: one_hot编码
    """
    # 每一类样本数，假设有1000个样本，分成两类，每类500个样本
    sample_per_class = int(sample_szie / num_classes)

    # 生成均值为mean，协方差为cov sample_per_class * len(mean)个样本，对应标签为0
    X0 = np.random.multivariate_normal(mean, cov, sample_per_class)  # 多变量正太分布
    Y0 = np.zeros(sample_per_class, dtype=np.int32)
    # 对于diff长度不够进行处理
    if len(diff) != num_classes - 1:
        tmp = np.zeros(num_classes - 1)
        tmp[0:len(diff)] = diff
        tmp[len(diff):] = diff[-1]
    else:
        tmp = diff

    for ci, d in enumerate(tmp):
        '''
        把list变成 索引-元素树，同时迭代索引和元素本身
        '''

        # 生成均值为mean+d,协方差为cov sample_per_class x len(mean)个样本 类别为ci+1
        X1 = np.random.multivariate_normal(mean + d, cov, sample_per_class)
        Y1 = (ci + 1) * np.ones(sample_per_class, dtype=np.int32)

        # 合并X0,X1  按列拼接
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if one_hot:
        Y0 = get_one_hot(Y0, num_classes)

        # 打乱顺序
    X, Y = shuffle(X0, Y0)

    return X, Y


if __name__ == "__main__":
    np.random.seed(10)
    num_features = 2
    num_samples = 100
    mean = np.random.rand(num_features)
    cov = np.eye(num_features)

    X, Y = generate(num_samples, mean, cov, [3.0], num_classes=2)

    colors = ['r' if l == 0 else 'b' for l in Y[:]]

    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("Scaled age")
    plt.ylabel("Tumor size")
    lab_dim = 1

    input_x = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, lab_dim])

    W = tf.Variable(tf.truncated_normal(shape=[num_features, lab_dim]), name='weight')
    b = tf.Variable(tf.zeros(shape=[lab_dim]), name='bias')

    out_put = tf.nn.sigmoid(tf.matmul(input_x, W) + b)
    print(out_put.get_shape().as_list())

    cost = tf.reduce_mean(-(input_y * tf.log(out_put) + (1 - input_y) * tf.log(1 - out_put)))

    loss = tf.reduce_mean(tf.square(input_y - out_put))

    train = tf.train.AdamOptimizer(0.04).minimize(cost)

    training_epochs = 50
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_data = list(zip(X, Y))

        for epoch in range(training_epochs):
            random.shuffle(train_data)
            mini_batchs = [train_data[k: k+batch_size] for k in range(0, num_samples, batch_size)]
            cost_list = []

            for mini_batch in mini_batchs:
                x_batch = np.asarray([x for x, y in mini_batch], dtype=np.float32).reshape(-1, num_features)
                y_batch = np.asarray([y for x, y in mini_batch], dtype=np.float32).reshape(-1, lab_dim)

                _, lossval = sess.run([train, loss], feed_dict={input_x: x_batch, input_y: y_batch})
                cost_list.append(lossval)

                print('Epoch {0} cost {1}'.format(epoch, np.mean(cost_list)))

        x = np.linspace(-1, 8, 200)
        y = -x*(sess.run(W)[0] / sess.run(W)[1]) - sess.run(b) / sess.run(W)[1]
        plt.plot(x, y, label='Fitteg Line')
        plt.legend()
        plt.show()






