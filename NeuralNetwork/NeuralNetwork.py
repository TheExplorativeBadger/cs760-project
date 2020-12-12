#Samantha Shrum 2020.12.10
#CS 760 Project Neural Network

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
import random

# Numpy array console printing options (for debugging)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=250)
np.set_printoptions(suppress = True)
np.set_printoptions(threshold = 20)

# Tensorflow backwards compatibility requirement
tf.compat.v1.disable_eager_execution()

##### Paramaters #####
# Input file name
filename = 'FinalCombinedFeaturesWithSeverity.csv'
# Ratio of training to testing data
validation_ratio = 0.1
# Number of nodes in the hidden layer
h_size = 20
# Initial weight distribution standard deviation
std_dev = 1
# Regularization magnitude
reg_mag = 0.1
# Threshold below which weights are zeroed out
zero_threshold = 0.005
# Total epochs
epochs = 200
# How many points to train on per epoch
epoch_length = 100
# Adam Optimizer learning rate
lrate = 0.0003
# Adam Optimizer epsilon
eps = 1e-8

##### Pre-Network #####
# Import dataset
data_in = np.genfromtxt(filename, delimiter = ';')
# Separate labels from features (labels : x, features : y)
all_x = data_in[:,2:]
all_y = data_in[:,0]

# Normalize features to [0,1]
all_x_min = np.amin(all_x, axis = 0)
all_x_max = np.amax(all_x, axis = 0)
all_x_norm = (all_x - all_x_min)/(all_x_max - all_x_min)

##### Define Network Methods #####
# Randomizes the initial weights
def init_weights(shape):

    weights = tf.random.normal(shape, stddev=std_dev)
    return tf.Variable(weights, trainable = True)

# Sigmoid Function
def apply_sigmoid(tensor):

    two = tf.scalar_mul(2,tf.nn.sigmoid(tensor))
    one = tf.ones(tf.shape(tensor),dtype=tf.float32)
    return tf.subtract(two,one)

# Runs an initial state through the network
def forwardprop(x, w_1, w_2):
    h    = apply_sigmoid(tf.matmul(x,w_1))
    yhat = tf.matmul(h,w_2)
    return yhat

# Zeros out sufficiently small weights
def zero_weights(weights):
    condition = tf.less(tf.multiply(weights,tf.sign(weights)), zero_threshold)
    new_weights = tf.where(condition, tf.zeros_like(weights), weights)
    return new_weights

# Select data from within the training set given a list of indices
def select_epoch_data(epoch_train_index, train_x, train_y):
	epoch_train_x = []
	epoch_train_y = []

	for i in epoch_train_index:

		epoch_train_x.append(train_x[i])
		epoch_train_y.append(train_y[i])

	return (np.array(epoch_train_x), np.array(epoch_train_y))

##### Network Training Loop #####
# Returns a trained network
# x : feature vectors, y : labels
def train_network(train_x, test_x, train_y, test_y):

    # Layer's sizes
    x_size = train_x.shape[1]   # Number of input nodes (based on input data)
    y_size = 1                  # Number of outcomes

    # Symbols
    x = tf.compat.v1.placeholder("float", shape=[None, x_size])
    y = tf.compat.v1.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Define Forward propagation
    predict = forwardprop(x, w_1, w_2)

    # Define Backward propagation
    w1_abs_array = tf.multiply(w_1, tf.sign(w_1))   # Magnitudes of input weights
    w2_abs_array = tf.multiply(w_2, tf.sign(w_2))   # Magnitudes of hidden weights
    regularization = (tf.reduce_sum(w1_abs_array)+tf.reduce_sum(w2_abs_array))*reg_mag  # Lasso Regularization term

    cost_array = tf.compat.v1.losses.mean_squared_error(labels=y, predictions=predict, reduction=tf.compat.v1.losses.Reduction.NONE)
    cost    = tf.reduce_mean(cost_array)*100 + regularization # Correctness + Regularization
    
    updates = tf.compat.v1.train.AdamOptimizer(learning_rate = lrate, epsilon = eps).minimize(loss = cost)

    # Initialize
    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Check training accuracy, testing accuracy, total cost, and regularization cost pre-training
    train_error = np.mean(np.abs(train_y - np.squeeze(sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]}))))
    test_error  = np.mean(np.abs(test_y - np.squeeze(sess.run(predict, feed_dict={x: test_x, y: test_y[:,np.newaxis]}))))
    cost_num = sess.run(cost,feed_dict={x: train_x, y: train_y[:,np.newaxis]})
    reg_num = sess.run(regularization,feed_dict={x: train_x, y: train_y[:,np.newaxis]})

    # Print information about the network pre-training
    print("Epoch = %d, train error = %.4f, test error = %.4f, cost = %.5f, reg = %.5f" % (0, train_error, test_error,cost_num,reg_num))

    # Train the network
    for epoch in range(epochs):

        # Select point indices for this epoch
        epoch_train_index = random.sample(range(len(train_x)),epoch_length)
        # Pull these indices from the full training set
        epoch_train_x, epoch_train_y = select_epoch_data(epoch_train_index, train_x, train_y)

        # Train with each training data point
        for j in range(epoch_length):
            # If epoch length is greater than training set size, wrap the index
            i = j % len(epoch_train_x)

            # Train network on individual training point
            sess.run(updates, feed_dict={x: epoch_train_x[i: i + 1], y: epoch_train_y[i: i + 1, np.newaxis]})

        # Zero out any weights that are sufficiently close to 0 (this promotes sparsity)
        w_1 = zero_weights(w_1)
        w_2 = zero_weights(w_2)

        # Check training accuracy, testing accuracy, cost, and regularization cost
        train_error = np.mean(np.abs(train_y - np.squeeze(sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]}))))
        test_error = np.mean(np.abs(test_y - np.squeeze(sess.run(predict, feed_dict={x: test_x, y: test_y[:,np.newaxis]}))))
        cost_num = sess.run(cost,feed_dict={x: train_x, y: train_y[:,np.newaxis]})
        reg_num = sess.run(regularization,feed_dict={x: train_x, y: train_y[:,np.newaxis]})

        # Print information about intermediate epochs to console
        print("Epoch = %d, train error = %.4f, test error = %.4f, total cost = %.5f, regularization cost = %.5f" % (epoch + 1, train_error, test_error,cost_num,reg_num))

    print("Training Complete.")

    # End training session
    sess.close()

    # Return trained weights
    return (w_1, w_2)

# Separate data into training set and testing set
train_x, test_x, train_y, test_y = train_test_split(all_x_norm, all_y, test_size=(validation_ratio))

# Run the network
train_network(train_x, test_x, train_y, test_y)
