# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/")

print('训练数据：', mnist.train.images)
print('训练数据shape', mnist.train.images.shape)

import pylab

im = mnist.train.images[1]
im = im.reshape(-1, 28)
pylab.imshow(im)
# pylab.show()

print('测试数据shape:', mnist.test.images.shape)
print('验证数据shape:', mnist.validation.images.shape)

import tensorflow as tf

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

z = tf.matmul(x, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.1

learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                           global_step=global_step,
                                           decay_rate=0.99,
                                           decay_steps=2000)
add_global = global_step.assign_add(1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1
decay_steps = 2000
total_batch = int(mnist.train.num_examples / batch_size)
print('total_batch', total_batch)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(learning_rate))

    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, s, lr = sess.run([optimizer, cost, add_global, learning_rate],
                                   feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch

            if s % decay_steps == 0:
                print('global step: %04d learning rate: %f' % (s, lr))

        if (epoch + 1) % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print(' Finished!')
