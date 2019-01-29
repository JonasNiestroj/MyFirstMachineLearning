import tensorflow as tf

tf.reset_default_graph()

x_scalar = tf.get_variable(
    'x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.get_variable('x_matrix', shape=[
                           30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

scalar_summary = tf.summary.scalar(name='Scalar_summary', tensor=x_scalar)
histogram_summary = tf.summary.histogram('Histogram_summary', x_matrix)

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(100):
        sess.run(init)
        summary = sess.run(merged)
        writer.add_summary(summary, step)
    print('Done')
