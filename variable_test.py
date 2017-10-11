import tensorflow as tf
x = tf.Variable([[1.0, 2.0], [1.1, 2.0]])
y = tf.Variable([[3.0, 4.0], [1.0, 3.0]])

#1 2   3 4
#2 2 x 1 3 =

init = tf.global_variables_initializer()

z = tf.matmul(x, y)

sess = tf.Session()
sess.run(init)

v = sess.run(x)
z = sess.run(z)
print(z)
