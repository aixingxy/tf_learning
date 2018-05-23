# coding=UTF-8
import tensorflow as tf
import os.path

MODEL_DIR = "model/ckpt"
MODEL_NAME = "model.ckpt"

# if os.path.exists(MODEL_DIR): 删除目录
#     shutil.rmtree(MODEL_DIR)
if not tf.gfile.Exists(MODEL_DIR):  # 创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

# 下面的过程你可以替换成CNN、RNN等你想做的训练过程，这里只是简单的一个计算公式
input_holder = tf.placeholder(tf.float32, shape=[1], name="input_holder")  # 输入占位符，并指定名字，后续模型读取可能会用的
W1 = tf.Variable(tf.constant(5.0, shape=[1]), name="W1")
B1 = tf.Variable(tf.constant(1.0, shape=[1]), name="B1")
_y = (input_holder * W1) + B1
predictions = tf.greater(_y, 50, name="predictions")  # 输出节点名字，后续模型读取会用到，比50大返回true，否则返回false

init = tf.global_variables_initializer()
saver = tf.train.Saver()  # 声明saver用于保存模型

with tf.Session() as sess:
    sess.run(init)
    print("predictions: ", sess.run(predictions, feed_dict={input_holder: [10.0]}))  # 输入一个数据测试一下)
    saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME))  # 模型保存
    print("%d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node))  # 得到当前图有几个操作节点

for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
    print(op.name, op.values())
