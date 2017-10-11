import tensorflow as tf
import numpy as np

a = tf.add(2, 3)
b = tf.multiply(a, 4)
c = tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(a))
    #v = sess.run(c)
    print(sess.run([c, a, b], feed_dict={a: 3, b: 5}))
print([1])
print(np.array([[2]]))
