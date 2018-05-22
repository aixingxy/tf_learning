# -*- coding:utf-8 -*-

'''
@author: xingxiaoyang

'''

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from tensorflow.python.tools import freeze_graph

learning_rate = 0.1

train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x * np.random.rand(100) * 0.3

with tf.variable_scope('place_holder'):
    X = tf.placeholder("float", name='input_placeholder')
    Y = tf.placeholder("float", name='label_placeholder')

with tf.variable_scope('NN'):
    W = tf.Variable(initial_value=tf.random_normal([1]), name="weight")
    b = tf.Variable(initial_value=tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b

with tf.variable_scope('cost'):
    cost = tf.reduce_mean(tf.square(Y - Z))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_epochs = 100
display_step = 10
saver = tf.train.Saver()

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    else:
        return [val if idx < w else sum(a[idx-w: w])/w for idx, val in enumerate(a)]

with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize": [], "loss": []}
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            saver.save(sess, "model/linearmodel", global_step=epoch)
            print("Epoch:", epoch+1, "cost:", loss, "W=", sess.run(W), "b=", sess.run(b))

            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Fisnished")
    print("cost:", loss, "W=", sess.run(W), "b=", sess.run(b))

    # plt.subplot(211)
    # plt.plot(train_x, train_y, 'ro', label="original data")
    # plt.plot(train_x, train_x * sess.run(W)+sess.run(b), label="FIttedline")
    # plt.legend()
    #
    #
    # plotdata["aveloss"] = moving_average(plotdata["loss"])
    #
    # plt.subplot(212)
    # plt.plot(plotdata["batchsize"], plotdata["aveloss"], 'b--')
    # plt.xlabel("minibatch number")
    # plt.ylabel("loss")
    # plt.show()
#
# epoch = 10
# with tf.Session() as sess2:
#     sess2.run(tf.global_variables_initializer())
#     saver.restore(sess2, "linearmodel-"+str(epoch))
#     # print('x=0.2, z=', sess2.run(Z, feed_dict={X: 0.2}))
#
#     tf.train.write_graph(sess2.graph_def, 'output_model/pb_model', 'model.pb')
#     freeze_graph.freeze_graph('output_model/pb_model/model.pb',
#                               '',
#                               False,
#                               "linearmodel-"+str(epoch),
#                               'out',
#                               'save/restore_all',
#                               'save/Const:0',
#                               'output_model/pb_model/frozen_model.pb',
#                               False,
#                               "")













