# Tensorflow图像处理
一张RGB色彩模式的图片可以看成一个三维矩阵，矩阵中的每一个数表示了图像上不同位置，不同颜色的亮度。然而图像再存储时并不是直接记录这些矩阵中的数字，
而是记录经过压缩编码之后的结果。所以要将一张图像还原成一个三维矩阵，需要解码的过程。Tensorflow提供了对jgep和png格式图像的编码和解码函数。

``` python
# 演示Tensorflow对jpeg格式图像的编码/解码函数
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("B.jpg", 'rb').read()

with tf.Session() as sess:
    # 将图像使用jpeg的格式解码从而得到图像对应的三维矩阵。Tensorflow还提供了tf.image.decode_png函数对png格式的图片进行解码。
    # 解码之后的结果为一个张量，在使用它的取值之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.eval())
    
    plt.imshow(img_data.eval())
    plt.show()

    # 转变图片数据类型，方便保存
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    
    # 将表示一张图片的三维矩阵重新按照jpeg格式编码并存入文件中。
    encode_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("read/test.jpg", 'wb') as f:
        f.write(encode_image.eval())

```
# 多线程输入数据处理框架

``` python
# -*- coding:utf-8 -*-
import tensorflow as tf

q = tf.FIFOQueue(capacity=2, dtypes=tf.int32)
# 使用enqueue_many函数来初始化队列中的元素。和变量初始化类似，在使用队列之前需要明确的调用这个初始化过程
init = q.enqueue_many(([0, 10], ))

# 使用Dequeue函数将队列中的第一个元素出队列，这个元素值将被存在变量x中
x = q.dequeue()
# 将得到的值加1
y = x + 1
# 将加1后的值重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 运行初始化队列的操作
    init.run()
    for _ in range(5):
        # 运行q_inc将执行数据出队列、出队元素+1、重新加入队列的整个过程
        v, _ = sess.run([x, q_inc])
        # 打印出队元素的取值
        print(v)
```

Tensorflow提供了tf.Coordinator和tf.QueueRunner两个类来完成多线程协同的功能。tf.Coordinator主要用于协同对个线程一起停止，
并提供should_stop、request_stop和join三个函数。在线程启动之前，需要先声明一个tf.Coordinator类，并将这个类传到每一个创建的线程中。
启动的线程需要一直查询tf.Coordinator类中提供的should_stop函数，当这个函数的返回值为True时，则当前线程也需要脱出。
每启动一个线程都可以通过调用request_stop函数来通知其他线程退出。当某一个线程调用request_stop函数之后，should_stop函数的返回值将被置为True，
这样其他的线程就可以同时终止了。
``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import threading
import time
# 线程中运行的程序，这个程序每隔1秒钟判断是否需要停止并打印自己的ID
def MyLoop(coord, worker_id):
    # 使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print("Stop from id: %d\n" % worker_id)
            # 调用coord.request_stop()函数来通知其他线程停止
            coord.request_stop()
        else:
            # 打印当前线程的ID
            print("Working on id: %d\n" % worker_id)
        # 暂停1秒
        time.sleep(1)
# 声明一个tf.train.Coordinator类来协同多个线程
coord = tf.train.Coordinator()
# 声明创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in xrange(5)]
# 启动所有线程
for t in threads:
    t.start()
# 等待所有线程退出
coord.join(threads)
```

tf.QueueRunner主要用于启动多个线程来操作同一个队列，启动的这些线程可以通过tf.Coordinator类来统一管理。
``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
# 声明一个先进先出的队列，队列中最对100个元素，类型为实数
queue = tf.FIFOQueue(100, tf.float32)
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])
# 使用tf.train.QueueRunner来创建多个线程运行队列的入队操作
# tf.train.QueueRunner的第一个参数给出了被操作的队列
# 第二个参数[enqueue_op]*5表示了需要启动5个线程，每个线程中运行的是enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)
# 将定义过的QueueRunner加入到Tensorflow计算图上指定的集合
# tf.train.add_queue_runner没有指定集合则加入默认集合tf.GraphKeys.QUEUE_RUNNERS
# 下面的操作就是讲刚刚定义的qr加入默认的tf.GraphKeys.QUEUE_RUNNERS集合
tf.train.add_queue_runner(qr)
#定义出队操作
out_tensor = queue.dequeue()
with tf.Session() as sess:
    # 使用tf.train.Coordinator来协同启动的线程
    coord = tf.train.Coordinator()
    # 使用tf.train.QueueRunner时，需要明确使用tf.train.start_queue_runners
    # 来启动所有线程。否则因为没有线程运行入队操作，当调用出队操作时，程序会一直等待入队操作被执行。
    # tf.train.start_queue_runner函数会模型启动tf.GraphKeys.QUEUE_RUNNERS集合中的所有QueueRunner。
    # 因为这个函数只支持启动指定集合找那个的QueueRunnner，所以一般涞水tf.add_queue_runner函数和
    # tf.train.start_queue_runner函数会指定同一个集合。
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 获取队列中的取值
    for _ in range(3):
        print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
```

# 输入文件队列
TensorFlow提供了tf.train.match_filenames_once函数来获取符合一个正则表达式的所有文件，
得到的文件列表可以通过tf.train.string_input_producer函数来进行有效的管理。


``` python
# -*- coding:utf-8 -*-
import tensorflow as tf
import glob
with tf.Session() as sess:
    # 创建文件列表
    file_name_list = glob.glob("./*.jpg")
    print(file_name_list)

    # 创建文件名队列
    file_queue = tf.train.string_input_producer(string_tensor=file_name_list, num_epochs=5, shuffle=False)

    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    # tf.train.string_input_producer中的num_epochsH和capacity是局部变量，所以需要使用局部变量初始化
    tf.local_variables_initializer().run()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    try:

        while not coord.should_stop():
            i += 1
            image = sess.run(value)
            with open("read/test_{}.jpg".format(i), 'wb') as f:
                f.write(image)
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")
    finally:
        # 运行结束，通知进程停止
        coord.request_stop()
    # 等待进程结束
    coord.join(threads)

```
