'''
python 3.5
实例1：从一组看似混乱的数据中找到y=2x的规律
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


train_x = np.linspace(-1, 1, 100)
train_y = 2 * train_x + np.random.rand(100) * 0.3

# plt.plot(train_x, train_y, 'ro', label='Original data')
# plt.legend()
# plt.show()


# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name='bias')

# 前向结构
z = tf.multiply(X, W) + b

# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 定义参数
training_epoch = 20
display_step = 2

#
saver = tf.train.Saver()

# 启动session
with tf.Session() as sess:
    sess.run(init)
    plot_data = {"batch_size": [], "loss": []}
    def moving_average(a, w=10):
        if len(a) < w:
            return a[:]
        return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
    # 向模型输入数据
    for epoch in range(training_epoch):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            print("Epoch：", epoch+1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plot_data["batch_size"].append(epoch)
                plot_data["loss"].append(loss)
    print(" Finished ")
    print("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W=", sess.run(W), "b=", sess.run(b))
    # 训练完之后，保存模型
    saver.save(sess, "log/model.ckpt")

# 图形显示
    plt.plot(train_x, train_y, 'ro', label="Origial data")
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label="FittedLine")
    plt.legend()
    plt.show()

    plot_data["avgloss"] = moving_average(plot_data["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batch_size"], plot_data["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs Training loss')
    plt.show()

# 模型的使用
saver = tf.train.Saver()
with tf.Session() as sess2:
    # 参数可以初始化，也可以不初始化。即使初始化了，初始化值还是会被restore的值覆盖
    saver.restore(sess2, "log/model.ckpt")
    print("x=0.2, z=", sess2.run(z, feed_dict={X: 0.2}))

