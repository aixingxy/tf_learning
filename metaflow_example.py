# -*-coding:utf-8 -*-
import tensorflow as tf
import os
#
dir = os.path.dirname(os.path.realpath(__file__))

# # 目前在默认graph中
#
# # 创建一些变脸
# v1 = tf.Variable(1, name="v1")
# v2 = tf.Variable(2, name='v2')
# # 创建一些操作
# a = tf.add(v1, v2, name='add')
#
# # 验证当前graph是否与默认图相同
# print(v1.graph == tf.get_default_graph())
#
# # 创建Saver
#
# all_saver = tf.train.Saver()
#
# v2_saver = tf.train.Saver({'v2': v2})
#
# with tf.Session() as sess:
#     # 初始化v1和v2
#     sess.run(tf.global_variables_initializer())
#
#     all_saver.save(sess, dir+'/data-all/data-all')
#     v2_saver.save(sess, dir+'/data-v2/data-v2')

#
# print(dir + "/data-all/data-all.meta")
saver = tf.train.import_meta_graph(dir + "/data-all/data-all.meta")

graph = tf.get_default_graph()

for op in tf.get_default_graph().get_operations():  # 打印模型节点信息
    print(op.name, '<--->', op.values())

v1_tensor = graph.get_tensor_by_name('v1:0')
v2_tensor = graph.get_tensor_by_name('v2:0')

with tf.Session() as sess:
    saver.restore(sess, dir +'/data-all/data-all')
    print(sess.run(v1_tensor))
    print(sess.run(v2_tensor))
