'''
Author: Trevor Rice
Course: CSC 546
Assignment: Homework 2: Create a deep and wide neural network that can determine
a number value(0-9) of a handwritten digit, using the MNIST data set. 
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

# loading the mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10 # 10 classes representing the numbers (0-9)

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])

# our output will be a number 0-9
Y = tf.placeholder(tf.float32, [None, nb_classes])

# first layer
nb_classes_layer1 = 40
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([784, nb_classes_layer1]))
    b1 = tf.Variable(tf.random_normal([nb_classes_layer1]))
    layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("W1", W1)
    b1_hist = tf.summary.histogram("b1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

# second layer
nb_classes_layer2 = 20
with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([nb_classes_layer1, nb_classes_layer2]))
    b2 = tf.Variable(tf.random_normal([nb_classes_layer2]))
    layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("W2", W2)
    b2_hist = tf.summary.histogram("b2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

# hypothesis
with tf.name_scope("output") as scope:
    W3 = tf.Variable(tf.random_normal([nb_classes_layer2, nb_classes]))
    b3 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

    w3_hist = tf.summary.histogram("W3", W3)
    b3_hist = tf.summary.histogram("b3", b3)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    cost_scalar = tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)

# testing the model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# calculating the accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# I found that roughly 60-80 epochs results in a very low cost with 1.5 learning rate
# if you wish to run fewer epochs to save time, lower this to around 30 epochs for a
# good balance of cost and time

num_epochs = 65
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # summary
    merged_summary = tf.summary.merge_all()

    # create the summary writer
    writer = tf.summary.FileWriter(os.getcwd() + '/logs/mnist_nn_1_5')
    # adding the graph
    writer.add_graph(sess.graph)

    # initialize global variables
    sess.run(tf.global_variables_initializer())
    # training cycles
    for epoch in range(num_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        # iterations
        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, summary, _ = sess.run([cost, merged_summary, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            # adding the summary to the writer
            writer.add_summary(summary, global_step=(num_iterations*epoch + i))
            avg_cost += c / total_batch
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
    print("Model has completed its learning")

    # Test the model using test sets
    print("Accuracy: ",accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    # get a random picture and predict its output using our model
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()

# runs the tensorboard command passing in the /logs directory for the logdir
os.system("tensorboard --logdir=" + os.getcwd() + '/logs')