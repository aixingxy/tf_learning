# 下载并显示MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
#
# print('输入数据：', mnist.train.images)
# print('输入数据的shape：', mnist.train.images.shape)
# import pylab
# im = mnist.train.images[1]
# im = im.reshape(-1, 28)
# pylab.imshow(im)
# pylab.show()

import tensorflow as tf
import matplotlib.pylab as plt

# 定义占位符
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
# tf.summary.image('image', tf.reshape(x, [-1, 28]))
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# tf.summary.scalar('label', y)

# 定义学习参数
W = tf.Variable(initial_value=tf.random_normal([784, 10]))
b = tf.Variable(initial_value=tf.zeros([10]))

# 定义输出
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
tf.summary.scalar('loss', cost)

# 定义学习率
learning_rate = 0.01

# 定义优化方法
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1

saver = tf.train.Saver()
model_path = "log/521model.ckpt"
# 启动session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     merged_summary_op = tf.summary.merge_all()
#     summary_writer = tf.summary.FileWriter('log/mnist_with_summaried', sess.graph)
#
#     for epoch in range(training_epochs):
#         avg_cost = 0
#         total_batch = int(mnist.train.num_examples/batch_size)
#         # 循环所有数据集
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
#
#             # 生成summary
#             summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
#             summary_writer.add_summary(summary_str, global_step=epoch)
#
#             avg_cost += c / total_batch
#         if (epoch + 1) % display_step == 0:
#             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
#     print("Finished!")
#
#     # 测试
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#
#     # 计算准确率
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
#
#     # 保存模型
#     save_path = saver.save(sess, model_path)
#     print("Model saved in file: %s" % save_path)

with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess2, model_path)
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess2.run([output, pred], feed_dict={x: batch_xs, y: batch_ys})
    # print(outputval, predv)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    plt.figure()
    plt.subplot(121)
    plt.imshow(im)
    # plt.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    plt.subplot(122)
    plt.imshow(im)
    plt.show()

