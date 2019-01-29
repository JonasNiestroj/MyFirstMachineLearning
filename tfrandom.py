import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, shape=[None, 784])
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[
                    784, 200], initializer=weight_initer)

bias_initer = tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

X_W = tf.matmul(X, W, name="MatMul")

X_W_b = tf.add(X_W, b, name="Add")

h = tf.nn.relu(X_W_b, name="ReLU")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    d = {X: np.random.rand(100, 784)}

    print(sess.run(h, feed_dict=d))
