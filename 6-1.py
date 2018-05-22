import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, 0.5, 6], [0.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)

result3 = -tf.reduce_sum(labels*tf.log(logits_scaled), 1)
with tf.Session() as sess:
    print("scaled = ", sess.run(logits_scaled), '\n')
    print("scaled sum = ", sess.run(tf.reduce_sum(logits_scaled, 1)), '\n')
    print("scaled2 =", sess.run(logits_scaled2), '\n')
    print("scaled2 sum =", sess.run(tf.reduce_sum(logits_scaled2, 1)), '\n')
    print("rel1 = ", sess.run(result1), '\n')
    print("relu2 = ", sess.run(result2), '\n')
    print("relu3 = ", sess.run(result3))