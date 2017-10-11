import tensorflow as tf
import pickle

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#with open('mnist_data', 'wb') as f:
#    pickle.dump(mnist, f)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 1

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


def train_neural_network(x, y):
    prediction = neural_network_model(x)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    #softmax = tf.nn.softmax(logits=prediction)
    sq_loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_images, epoch_labels = mnist.train.next_batch(batch_size)
        res = sess.run(prediction, feed_dict={x: epoch_images})
        res1 = sess.run(softmax, feed_dict={x: epoch_images, y: epoch_labels})
        sq_loss1 = sess.run(sq_loss, feed_dict={x: epoch_images, y: epoch_labels})

        print(res)
        print(res1)
        print(sq_loss1)
        print(epoch_labels)


train_neural_network(x, y)
