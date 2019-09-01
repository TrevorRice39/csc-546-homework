import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np


# os.getcwd() gets the current working directory
data_path = os.getcwd() + '/data.csv' 

# loading the csv using numpy.loadtxt
xy = np.loadtxt(data_path, delimiter=',', dtype=np.float32)

# slicing out all columns except the last column for x_data
x_data = xy[:, 0:-1]

# slicing out the last column for y_data
y_data = xy[:, [-1]]


# checking the shape of the data
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# place holder for our x_data
X = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
 
# place holder for our y_data
Y = tf.placeholder(tf.float32, shape=[None, 1])


W = tf.Variable(tf.random_normal([x_data.shape[1],1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = { X: x_data, Y: y_data})
    print(step, "cost:", cost_val, "\nprediction:", hy_val)