#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:18:34 2018
@author: xingxiaoyang
"""

# 用于通过读取图片的path,然后解码成图片数组的形式，最后返回batch个图片数组
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

img_path = glob.glob('*.jpg')
img_path = tf.convert_to_tensor(img_path, dtype=tf.string)

image = tf.train.slice_input_producer([img_path], num_epochs=10, shuffle=False)


def load_img(path_queue):
    file_contents = tf.read_file(Z[0])
    image = tf.image.decode_jpeg(file_contents, channels=3)

    image = tf.image.resize_images(image, size=(200, 200))
    tf.image.convert_image_dtype(image, dtype=tf.uint8)

    return image


img = load_img(image)
print(img.shape)

image_batch = tf.train.batch([img], batch_size=10)

with tf.Session() as sess:
    # initializer for num_epochs
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    try:
        while not coord.should_stop():
            imgs = sess.run(image_batch)

            for j in range(imgs.shape[0]):
                img_slice = imgs[j, :, :, :]
                print(img_slice.shape)
                encode_image = tf.image.encode_jpeg(img_slice)
                with tf.gfile.GFile("read/test_%d.jpg" % i, 'wb') as f:
                    f.write(encode_image.eval())
                    i += 1

            print(imgs.shape)
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(thread)

# import glob
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
#
# image_list = glob.glob('*.jpg')
# label_list = [1, 2, 3, 4]
# image_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
# label_tensor = tf.convert_to_tensor(label_list, dtype=tf.uint8)
#
# file = tf.train.slice_input_producer([image_tensor, label_tensor])
#
# image_content = tf.read_file(file[0])
# index = file[1]
#
# image_data = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_content), tf.uint8)
#
# with tf.Session() as sess:
#     sess.run(tf.local_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     try:
#         if not coord.should_stop():
#             sess.run(image_data)
#
#             print(sess.run(index))
#     except tf.errors.OutOfRangeError:
#         print("Done")
#     finally:
#         coord.request_stop()
#     coord.join(threads)
#     plt.imshow(image_data.eval())
#     plt.show()
