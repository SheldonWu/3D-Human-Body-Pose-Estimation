'''
A Multilayer Perceptron implementation example using TensorFlow library.

Based off example from Aymeric Damien (https://github.com/aymericdamien/TensorFlow-Examples/)

Copyright 2017
Simon Kalouche
'''

from __future__ import print_function
from parseData import getParams, getEgoCapData, getOneExample
import tensorflow as tf
from ipdb import set_trace as debug
import numpy as np

def normalize_data(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)

# Create model
def multilayer_perceptron(x, weights, biases, dropout=1):

    num_layers = len(weights.keys()) - 1
    layer = dict()
    layer[0] = x

    for i in range(num_layers):
        # Hidden layer with RELU activation
        layer[i+1] = tf.add(tf.matmul(layer[i], weights['h' + str(i+1)]), biases['b' + str(i+1)])         # affine (fully connected) layer
        layer[i+1] = tf.contrib.layers.batch_norm(layer[i+1])        
        layer[i+1] = tf.nn.relu(layer[i+1])
        layer[i+1] = tf.nn.dropout(layer[i+1], keep_prob=dropout)

    # Output layer with linear activation
    out_layer = tf.matmul(layer[num_layers], weights['out']) + biases['out']
    return out_layer


def train(learning_rate, training_epochs, batch_size, dropout, n_hidden, model_name='model', save=True):
    # Train the neural network 

    # get all the params and data in one go. Make sure these are in the right order (just copy from output of setupNeuralNet if not sure)
    EgoCap, num_examples, display_step, normalize, n_input, n_classes, \
    num_layers, weights, biases = setupNeuralNet(n_hidden=n_hidden, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, dropout=dropout)

    # normalize the data
    if normalize:
        X_train = normalize_data(EgoCap.X_train)
        X_test = normalize_data(EgoCap.X_test)
        y_train = normalize_data(EgoCap.y_train)
        y_test = normalize_data(EgoCap.y_test)
    else:
        X_train = EgoCap.X_train
        X_test = EgoCap.X_test
        y_train = EgoCap.y_train
        y_test = EgoCap.y_test

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    logger = np.zeros((training_epochs,3))

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, dropout)

    # Define loss and optimizer
    cost = tf.reduce_mean((pred - y)**2)        # L2 loss for regression, softmax for classification: (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # adds ops to save and restore all the variables
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(num_examples/batch_size)

            # Loop over all batches
            for i in range(total_batch):
                indices = range(i*batch_size, (i+1)*batch_size)
                batch_x, batch_y = X_train[indices], y_train[indices]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:
                # calculate
                rmse = tf.sqrt(tf.reduce_mean(tf.square(pred - y)))

                logger[epoch] = [avg_cost, rmse.eval({x: X_train, y: y_train}), rmse.eval({x: X_test, y: y_test})]

                print("Epoch:", (epoch+1), 
                      "loss=", avg_cost,
                      "rmse_train=", logger[epoch,1], 
                      "rmse_test=", logger[epoch,2])
        print("Optimization Finished!")

        # Test model Classification
        # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))    # for classification
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))   # for classification
        
        # Calculate accuracy
        correct_prediction = tf.abs(pred - y)   # absolute difference between the prediction and the label
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), axis=0) # take the average across each joint
        rmse = tf.sqrt(tf.reduce_mean(tf.square(pred - y), axis=0))

        # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
        print("RMSE:", rmse.eval({x: X_test, y: y_test}))
        print("RMSE Train:", rmse.eval({x: X_train, y: y_train}))


        # save the weights mofo
        if save:
            file_ID = '_numLayers' + str(num_layers) + '_epochs' + str(training_epochs) + '_drop' + str(dropout) + '_lr' + str(learning_rate)
            
            # Save the variables to disk.
            weights_file = 'weights/weights' + file_ID
            save_path = saver.save(sess, "/tmp/" + model_name + ".ckpt")
            print("Model saved in file: %s" % save_path)

            # bias_file = 'weights/bias' + file_ID
            # np.save(weights_file, biases)

            results_file = 'results/trials/log' + file_ID
            np.save(results_file, logger)

            rmse_file = 'results/trials/rmse' + file_ID
            np.save(rmse_file, rmse.eval({x: X_test, y: y_test}))

            # save as csv
            # np.savetxt('results/trials/log' + file_ID + '.csv', logger, delimiter=",")
            # np.savetxt('results/trials/rmse' + file_ID + '.csv', rmse.eval({x: X_test, y: y_test}), delimiter=",")



def makePrediction(pixelsXY, n_hidden, model_name='model'):
    # make a prediction on a single example given a (36,1) shape input
    EgoCap, num_examples, display_step, normalize, n_input, n_classes, \
    num_layers, weights, biases = setupNeuralNet(n_hidden=n_hidden)

    # define varaibles
    x = tf.placeholder("float", [None, n_input])  

    # prediction step
    pred = multilayer_perceptron(x, weights, biases, dropout=1)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "/tmp/" + model_name + ".ckpt")
        print("Model restored.")

        out = pred.eval({x: pixelsXY})

    return out


def setupNeuralNet(n_hidden, learning_rate=.001, training_epochs=12, batch_size=64, dropout=1):
    # Import EgoCap Data
    EgoCap = getEgoCapData()

    # get the EgoCap params 
    train_params = getParams(mode='train')
    test_params = getParams(mode='test')
    num_joints = train_params.num_body_parts
    num_examples = EgoCap.X_train.shape[0]

    # Parameters
    display_step = 1
    normalize = False

    # Network Parameters
    # n_hidden = number of features for each hidden layer. len(n_hidden) = number of hidden layers
    n_input = num_joints*2      # EgoCap 2D Labels (num_joints * 2)
    n_classes = num_joints*3    # Egocap 3D labels (num_joints*3)
    dimens = list([n_input] + n_hidden + [n_classes]) 
    num_layers = len(dimens) - 1    # subtract 1 because input doesnt count as a layer
    weights, biases = dict(), dict()

    # store and initialize an arbitrary number of hidden layers weights and biases using a loop 
    for i in range(num_layers):
        if i >= num_layers - 1:
            weights['out'] = tf.Variable(tf.random_normal([dimens[i], dimens[i+1]]))
            biases['out'] = tf.Variable(tf.random_normal([dimens[i+1]]))
        else:
            weights['h' + str(i+1)] = tf.Variable(tf.random_normal([dimens[i], dimens[i+1]]))
            biases['b' + str(i+1)] = tf.Variable(tf.random_normal([dimens[i+1]]))

    params = [EgoCap, num_examples, display_step, normalize, n_input, n_classes, num_layers, weights, biases]

    return params



if __name__ == "__main__":

    # set run mode
    mode = 'train'
    model_name = 'model_2'

    if mode == 'train':
        train(n_hidden=[256, 256, 1024, 1024], learning_rate=.001, training_epochs=12000, batch_size=64, dropout=.6, model_name=model_name, save=True)

    elif mode == 'test':
        # Just print 3D coordinate prediction for 1 example
        example_ID = 3922           # 4586 is the ID number for S7_v003_cam0_frame-3922.jpg taken from validation_v003_2D/dataset.txt   
        prediction = makePrediction(getOneExample(example_ID), model_name=model_name, n_hidden=[1024, 1024])
        print("Example Prediction: \n", prediction) 



