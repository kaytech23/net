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

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.placeholder('float', [None, 28 * 28])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input data * weights) + biases -> activation

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  W1 = tf.Variable(tf.random_normal([784, 500]))
  b1 = tf.Variable(tf.random_normal([500]))
  y1 = tf.add(tf.matmul(x, W1), b1)
  y1 = tf.nn.relu(y1)

  # W2 = tf.Variable(tf.random_normal([500, 500]))
  # b2 = tf.Variable(tf.random_normal([500]))
  # y2 = tf.add(tf.matmul(y1, W2), b2)
  # y2 = tf.nn.relu(y2)
  #
  # W3 = tf.Variable(tf.random_normal([500, 500]))
  # b3 = tf.Variable(tf.random_normal([500]))
  # y3 = tf.add(tf.matmul(y2, W3), b3)
  # y3 = tf.nn.relu(y3)

  W4 = tf.Variable(tf.random_normal([500, 10]))
  b4 = tf.Variable(tf.random_normal([10]))
  y = tf.add(tf.matmul(y1, W4), b4)
  #
  y_ = tf.placeholder(tf.float32)
  #y = neural_network_model(x)


  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  #train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

  #sess = tf.InteractiveSession()
  #tf.global_variables_initializer().run()
  # Train
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(25):
      epoch_loss = 0
      print("start: ", epoch_loss)
      for _ in range(int(mnist.train.num_examples / 100)):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        epoch_loss += c
#        print("epoch loss __: ", epoch_loss)
      print("epoch loss: ", epoch_loss)
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