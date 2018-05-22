# -*- coding:utf-8 -*-
import tensorflow as tf
import glob
with tf.Session() as sess:
    # 创建文件列表
    file_name_list = glob.glob("./*.jpg")
    print(file_name_list)

    # 创建文件名队列
    file_queue = tf.train.string_input_producer(string_tensor=file_name_list, num_epochs=5, shuffle=False)
    # file_queue = tf.train.slice_input_producer()

    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)

    tf.local_variables_initializer().run()
    # sess.run(tf.global_variables_initializer())
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


