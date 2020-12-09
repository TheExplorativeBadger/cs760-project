#Samantha Shrum 2020.11.23
#CS 760 Project Neural Network

import warnings 
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import tensorflow as tf
import sys
import numpy as np
import sklearn

from sklearn import datasets
from sklearn.model_selection import train_test_split
import random
from math import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Numpy array console printing options (for debugging)
np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=8)
np.set_printoptions(linewidth=250)
np.set_printoptions(suppress = True)

tf.compat.v1.disable_eager_execution()

##### Paramaters #####
#Input file name
filename = 'FinalCombinedFeaturesWithSeverity.csv'

#Ratio of training to testing data
validation_ratio = 0.1
#Number of nodes in the hidden layer
h_size = 6
#Initial weight distribution standard deviation
std_dev = 1
#Regularization magnitude
reg_mag = 0.02
zero_threshold = 0.005
#
epochs = 50
epoch_length = 500
#
lrate = 0.01
eps = 1e-8

##### Pre-Network #####
#import dataset
data_in = np.genfromtxt(filename, delimiter = ';')

#separate labels from features (labels : x, features : y)
all_x = data_in[:,1:]
all_y = data_in[:,0]

##### Define Network Methods #####
#Randomizes the initial weights
def init_weights(shape):
    #Weight initialization
    weights = tf.random.normal(shape, stddev=std_dev)

    return tf.Variable(weights, trainable = True)

def apply_sigmoid(tensor):

    two = tf.scalar_mul(2,tf.nn.sigmoid(tensor))
    one = tf.ones(tf.shape(tensor),dtype=tf.float32)
    return tf.subtract(two,one)

#Runs an initial state through the network
def forwardprop(x, w_1, w_2):
    h    = apply_sigmoid(tf.matmul(x,w_1))
    yhat = tf.matmul(h,w_2)
    return yhat

def zero_weights(weights):
    condition = tf.less(tf.multiply(weights,tf.sign(weights)), zero_threshold)
    new_weights = tf.where(condition, tf.zeros_like(weights), weights)
    return new_weights

def selectEpochData(epoch_train_index, train_x, train_y):
	epoch_train_x = []
	epoch_train_y = []

	for i in epoch_train_index:

		epoch_train_x.append(train_x[i])
		epoch_train_y.append(train_y[i])

	return (np.array(epoch_train_x), np.array(epoch_train_y))

##### Network Training Loop #####
#Creates the base framework of the network
def runNetwork():

	#Separates data into training set and testing set
    train_x, test_x, train_y, test_y = train_test_split(all_x, all_y, test_size=(validation_ratio))

    #Layer's sizes
    x_size = train_x.shape[1]   # Number of input nodes (based on input data)
    y_size = 1                  # Number of outcomes

    #Symbols
    x = tf.compat.v1.placeholder("float", shape=[None, x_size])
    y = tf.compat.v1.placeholder("float", shape=[None, y_size])

    #Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    #Define Forward propagation
    predict = forwardprop(x, w_1, w_2)

    #Define Backward propagation
    w1_abs_array = tf.multiply(w_1, tf.sign(w_1))
    w2_abs_array = tf.multiply(w_2, tf.sign(w_2))
    regularization = (tf.reduce_sum(w1_abs_array)+tf.reduce_sum(w2_abs_array))*reg_mag
    
    #yhat_norm = tf.scalar_mul(0.5,tf.add(apply_sigmoid(yhat),tf.ones(tf.shape(yhat))))

    cost_array = tf.compat.v1.losses.absolute_difference(labels=y, predictions=predict, reduction=tf.compat.v1.losses.Reduction.NONE)
    cost    = tf.reduce_mean(cost_array)*100 + regularization# + stupid_cost
    
    updates = tf.compat.v1.train.AdamOptimizer(learning_rate = lrate, epsilon = eps).minimize(loss = cost)

    #testing = yhat_norm

    #Run Stochastic Gradient Descent
    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    #Store all weights for analysis
    w1List = []
    w2List = []
    metadata = []

    # Check training accuracy, testing accuracy, and cost
    print(np.squeeze(sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]})))
    print(train_y)
    print(train_y - np.squeeze(sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]})))
    #print(np.absolute(np.squeeze(train_y) - sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]})))
    train_accuracy = np.mean(np.absolute(np.squeeze(train_y) - sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]})))
    test_accuracy  = np.mean(np.absolute(np.squeeze(test_y) - sess.run(predict, feed_dict={x: test_x, y: test_y[:,np.newaxis]})))
    cost_num = sess.run(cost,feed_dict={x: train_x, y: train_y[:,np.newaxis]})
    reg_num = sess.run(regularization,feed_dict={x: train_x, y: train_y[:,np.newaxis]})

    w1List.append(w_1.eval())
    w2List.append(w_2.eval())
    metadata.append([0,train_accuracy,test_accuracy,cost_num,reg_num])

    # Print information about intermediate epochs to console
    print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%, cost = %.5f, reg = %.5f" % (0, 100. * train_accuracy, 100. * test_accuracy,cost_num,reg_num))

    step = 0

    # Train the network
    for epoch in range(epochs):

        #print(sess.run(cost_array, feed_dict={x: train_x, y: train_y}))

        #create training set for this epoch
        #select point indices
        epoch_train_index = random.sample(range(len(train_x)),epoch_length)
        #pull these indices from the full training set
        epoch_train_x, epoch_train_y = selectEpochData(epoch_train_index, train_x, train_y)


        # Train with each training data point
        for j in range(epoch_length):
            step = step + 1
            '''
            if write_to_file == True:
                if step % record_step == 0:
                    w1List.append(w_1.eval())
                    w2List.append(w_2.eval())
                    metadata.append([epoch,train_accuracy,test_accuracy,cost_num,reg_num])
            '''
            i = j % len(train_x)

            #Train network on individual training point
            sess.run(updates, feed_dict={x: epoch_train_x[i: i + 1], y: epoch_train_y[i: i + 1, np.newaxis]})
            #print(sess.run(predict, feed_dict={x: epoch_train_x[i: i + 1], y: epoch_train_y[i: i + 1, np.newaxis]}), epoch_train_y[i: i + 1, np.newaxis])

        w_1 = zero_weights(w_1)
        w_2 = zero_weights(w_2)

        # Check training accuracy, testing accuracy, and cost
        train_accuracy = np.mean(np.squeeze(train_y) - sess.run(predict, feed_dict={x: train_x, y: train_y[:,np.newaxis]}))
        test_accuracy  = np.mean(np.squeeze(test_y) - sess.run(predict, feed_dict={x: test_x, y: test_y[:,np.newaxis]}))
        cost_num = sess.run(cost,feed_dict={x: train_x, y: train_y[:,np.newaxis]})
        reg_num = sess.run(regularization,feed_dict={x: train_x, y: train_y[:,np.newaxis]})

        last_epoch = epoch + 1
        last_train = train_accuracy
        last_test = test_accuracy
        last_cost = cost_num
        last_reg = reg_num

        # Print information about intermediate epochs to console
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%, cost = %.5f, reg = %.5f" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy,cost_num,reg_num))

    final_out = ''
    final_out += "Epoch = {}. ".format(last_epoch)
    final_out += "Train Accuracy = %.2f%%. " % (100. * last_train.item())
    final_out += "Test Accuracy = %.2f%%. " % (100. * last_test.item())
    final_out += "Cost = %.5f. " % (last_cost.item())
    final_out += "Reg = %.5f. " % (last_reg.item())

    final_out = "Training Complete.\n" + final_out
    print(final_out)

    #End training session
    sess.close()

runNetwork()
