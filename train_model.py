
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tensorflow.python.framework import ops
import math as math
from PIL import Image


# read in our processed dataset of spectrograms and popularity classes
processed_dataset = pd.read_csv('processed_dataset.csv')
processed_dataset = processed_dataset.values
# print(processed_dataset.shape)
image_list = []

#build matrix of spectrograms
for item in processed_dataset:
    #read in image, convert to matrix of pixels, chop off alpha values, convert to gray-scale
    image_file = item[3]
    image = imageio.imread(image_file)
    image = np.dot(image[...,:4], [0.299, 0.587, 0.114, 0])
    image_list.append(image)

#print(len(image_list))
image_data = np.stack(image_list, axis=0)
#print(image_data.shape)
labels_data = processed_dataset[:, 2]

# Create train, dev, and test sets. Split into 80%, 10%, 10% respectively
train_data_end_index = int(image_data.shape[0] * 0.8)
dev_data_end_index = int(image_data.shape[0] * 0.9)

train_data = image_data[0:train_data_end_index]
dev_data = image_data[train_data_end_index:dev_data_end_index]
test_data = image_data[dev_data_end_index:]

train_labels = labels_data[0:train_data_end_index]
dev_labels = labels_data[train_data_end_index:dev_data_end_index]
test_labels = labels_data[dev_data_end_index:]

train_labels = np.eye(3)[train_labels.astype(int)]
dev_labels = np.eye(3)[dev_labels.astype(int)]
test_labels = np.eye(3)[test_labels.astype(int)]

train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
dev_data = np.reshape(dev_data, (dev_data.shape[0], dev_data.shape[1], dev_data.shape[2], 1))
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1],test_data.shape[2], 1))

# create tf tensors for model
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    # x will contain train examples
    features = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name='x')
    # labels will contain the true popularity scores
    labels = tf.placeholder(tf.float32, shape=[None, n_y], name='y_true')
    return features, labels

X, Y = create_placeholders(train_data.shape[1], train_data.shape[2], 1, 3)
#print ("X =" + str(X))
#print ("Y =" + str(Y))

def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [3, 5, 1, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3, 5, 64, 128], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [2, 2, 128, 256], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [2, 2, 256, 512], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1":W1,
                  "W2":W2,
                  "W3":W3,
                  "W4":W4}
    
    return parameters

def forward_propogation(X, parameters, training):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    
    # first convolutional layer:     strides = 1 ... [1,s,s,1]
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding='VALID')
    # relu activation
    A1 = tf.nn.relu(Z1)
    # maxpool:     strides = 2... [1,s,s,1], window = 2x2...[1,f,f,1]
    P1 = tf.nn.max_pool(A1, strides = [1,2,2,1], ksize=[1,2,2,1], padding='VALID')

    # second convolutional layers:     strides = 1 ... [1,s,s,1]
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding='VALID')
    # relu activation
    A2 = tf.nn.relu(Z2)
    # maxpool:     strides = 2... [1,s,s,1], window = 2x2...[1,f,f,1]
    P2 = tf.nn.max_pool(A2, strides=[1,2,2,1], ksize=[1,2,2,1], padding='VALID')
    
    # third convolutional layer:     strides = 1 ... [1,s,s,1]
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding='VALID')
    # relu activation
    A3 = tf.nn.relu(Z3)
    # maxpool:     strides = 2... [1,s,s,1], window = 2x2...[1,f,f,1]
    P3 = tf.nn.max_pool(A3, strides = [1,2,2,1], ksize=[1,2,2,1], padding='VALID')
    
    # fourth convolutional layer:     strides = 1 ... [1,s,s,1]
    Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding='VALID')
    # relu activation
    A4 = tf.nn.relu(Z4)
    # maxpool:     strides = 2... [1,s,s,1], window = 2x2...[1,f,f,1]
    P4 = tf.nn.max_pool(A4, strides = [1,2,2,1], ksize=[1,2,2,1], padding='VALID')
    
    
    
    #Flatten
    P5 = tf.contrib.layers.flatten(P4)
    
    # fully connected layer with dropout
    fc_1 = tf.layers.dense(inputs=P5, units=1024, activation=tf.nn.relu)
    fc_1_dropout = tf.layers.dropout(
      inputs=fc_1, rate=0.6, training=training)
    
    Z5 = tf.contrib.layers.fully_connected(fc_1_dropout, 3, activation_fn=tf.nn.softmax)
    
    return Z5

def compute_cost(Z5, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	m = X.shape[0]
	mini_batches = []
	np.random.seed(seed)

	# Step 1:
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size)
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00005,
          num_epochs = 1, minibatch_size = 16, print_cost = True):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z5 = forward_propogation(X, parameters, True)
    
    cost = compute_cost(Z5,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):
            
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
            
            if print_cost == True and epoch % 1 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        # plot the cost 
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('costs_plot.png')
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        predict_op = tf.argmax(Z5, 1)
#         print("here1")
        correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))
#         print("here2")
            
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        tf.train.start_queue_runners(sess)
#         print("new here")
#         train_accuracy = 0
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
#         print("here3")
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
#         print("here4")

        print("Train Accuracy", train_accuracy)
        print("Test Accuracy", test_accuracy)
            
        return train_accuracy, test_accuracy, parameters

_, _, parameters = model(train_data, train_labels, dev_data, dev_labels)
