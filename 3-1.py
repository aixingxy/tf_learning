import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# 定义生成loss可视化的函数
plot_data = {"batch_size": [], "loss": []}

def moving_average(a, w=10):
    if len(a) < 10:
        return a[:]
    return [val if idx < 10 else sum(a[idx-10: idx])/w for idx, val in enumerate(a)]

# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.rand(100) * 0.3

# 图像显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
# plt.show()

tf.reset_default_graph()

# 创建模型
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 向前结构
z = tf.multiply(X, W) + b
tf.summary.histogram('z', z)

# 反向优化
cost = tf.reduce_mean(tf.square(Y-z))
tf.summary.scalar('loss_function', cost)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 定义学习参数
training_epoch = 200
display_stp = 2
saver = tf.train.Saver(max_to_keep=1) # 最多保存多少个ckpt，默认是5个
save_dir = "log/"

# 启动图
with tf.Session() as sess:

    sess.run(init)
    merged_summary_op = tf.summary.merge_all() # 合并所有summary

    # 创建summary_writer，用于写文件
    summary_writer = tf.summary.FileWriter('log/linear_summaries', sess.graph)

    # 向模型输入数据
    for epoch in range(training_epoch):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str, epoch) # 将summary写入文件

        # 显示训练详情
        if epoch % display_stp == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("epoch:", epoch+1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == "NA"):
                plot_data["batch_size"].append(epoch)
                plot_data["loss"].append(loss)
            saver.save(sess, save_dir+"linear_model.ckpt", global_step=epoch)

    print("Finished")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    # 显示模型
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label="Fitted Wline")
    plt.legend()
    # plt.show()

    plot_data["avg_loss"] = moving_average(plot_data["loss"])
    plt.figure()
    plt.subplot(211)
    plt.plot(plot_data["batch_size"], plot_data["avg_loss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibath run VS. Training loss")
    # plt.show()

# 重启一个session，载入检查点
load_epoch = 18
with tf.Session() as sess2:
# 方法1：同指定迭代次数
#     sess2.run(tf.global_variables_initializer())
#     saver.restore(sess2, save_dir+"linear_model.ckpt-"+str(load_epoch))
#     print("x = 0.2, z =", sess2.run(z, feed_dict={X: 0.2}))

# 方法2：使用 get_checkpoint_file(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess2, ckpt.model_checkpoint_path)

# 方法3：使用last_checkpoint
#     ckpt = tf.train.latest_checkpoint(save_dir)
#     if ckpt != None:
#         saver.restore(sess2, ckpt)

# 打印模型参数
# with tf.Session() as sess3:
#     sess3.run(tf.global_variables_initializer())
#     saver.restore(sess3, save_dir+"linear_model.ckpt-"+str(load_epoch))
#     print_tensors_in_checkpoint_file(save_dir+"linear_model.ckpt-"+str(load_epoch), None, True)


