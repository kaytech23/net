import tensorflow as tf


v = tf.Variable([1, 2])
init = tf.global_variables_initializer()

x = tf.Variable([4])

z = tf.multiply(3, x)

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(z))
    # sess.run(init)
    # Usage passing the session explicitly.
    # print(v.eval(sess))
    # Usage with the default session.  The 'with' block
    # above makes 'sess' the default session.
    # print(v.eval())
