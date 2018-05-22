'''
建立一个Session，打印hello
'''

# import tensorflow as tf
# hello = tf.constant('hello')
# with tf.Session() as sess:
#     print(sess.run(hello))


#*************************#

# a = tf.constant(3)
# b = tf.constant(2)
# with tf.Session() as sess:
#     print("相加: %i" % sess.run(a+b))
#     print("相乘: %i" % sess.run(a*b))

#*************************#
'''
演示注入机制，将具体的实参注入到相应的placeholder中。
feed只在调用她的方法内有效，方法结束后feed就会消失
'''
# import tensorflow as tf
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)
#
# add = tf.add(a, b)
# mul = tf.multiply(a, b)
#
#
# with tf.Session() as sess:
#     print("相加: %i" % sess.run(add, feed_dict={a: 3, b: 2}))
#     print("相乘: %i" % sess.run(mul, feed_dict={a: 3, b: 2}))
#     print(sess.run([mul, add], feed_dict={a: 3, b: 2}))
'''
采用feed机制可以在训练的时候每次喂如不同的数据，而采用constant定义参数就固定了，训练的时候就不能修改了。
'''
#*************************#
'''
详细介绍模型保存的细节：
模型虽然被保存到了log文件夹下，但是仍然是不透明的，
通过编写代码将模型里面的内容打印出来
'''

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"
print_tensors_in_checkpoint_file(savedir+"model.ckpt", None, True)

# tensor_name:  weight
# [1.9714612]
# tensor_name:  bias
# [0.14546445]

#*************************#
'''
指定存储变量的名字
'''
import tensorflow as tf
W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name="bias")

saver = tf.train.Saver({"weight": b, "bias": W})

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver.save(sess, "log/model1.ckpt")
print_tensors_in_checkpoint_file("log/model1.ckpt", None, True)

# tensor_name:  weight
# 2.0
# tensor_name:  bias
# 1.0
# 在创建saver时将weight和bias颠倒了