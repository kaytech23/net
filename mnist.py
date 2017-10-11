# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    stddev = 0.5
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W1 = tf.Variable(tf.random_normal([784, 1500], stddev=stddev))
    b1 = tf.Variable(tf.random_normal([1500], stddev=stddev))
    y1 = tf.matmul(x, W1) + b1
    y1 = tf.nn.relu(y1)

    W2 = tf.Variable(tf.random_normal([1500, 10], stddev=stddev))
    b2 = tf.Variable(tf.random_normal([10], stddev=stddev))
    y = tf.matmul(y1, W2) + b2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    # https://github.com/tensorflow/tensorflow/issues/2462
    # https://www.youtube.com/watch?v=G8eNWzxOgqE&list=PLAwxTw4SYaPn_OWPFT9ulXLuQrImzHfOV&index=7
    # softmax = tf.nn.softmax(y)
    #y_div = tf.scalar_mul(1/10, y)
    softmax = tf.nn.softmax(y)
    arr_crossentropy = y_ * tf.log(softmax)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(softmax), reduction_indices=[1]))
    logsoftmax = tf.log(softmax)
    # logits = y_ * tf.log(y);
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(softmax)))


    # cross_entropy = tf.reduce_mean(
    #    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(1)
        # _, l = sess.run([train_step, y], feed_dict={x: batch_xs, y_: batch_ys})
        ac, s, c, l, ys = sess.run([arr_crossentropy, logsoftmax, cross_entropy, softmax, y], feed_dict={x: batch_xs, y_: batch_ys})
        print("logits / scores")
        print(ys)

        print("probabilities after softmax")
        print(l)

        print("log softmax")
        print(s)

        print("cross entropy per ")
        print(ac)

        print("cross")
        print(c)

        break

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)