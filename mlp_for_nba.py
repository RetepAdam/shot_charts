'''
Code implements multi-perceptron neural network to classify MNIST images of
handwritten digits using Keras and Theano.  Based on code from
https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras

Note: neural network geometry not optimized (accuracy could be much better!)
'''

from __future__ import division
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano

def load_and_condition_MNIST_data():
    ''' loads and shapes MNIST image data '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print "\nLoaded MNIST images"
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    X_train.resize(len(y_train), 784) # 28 pix x 28 pix = 784 pixels
    X_test.resize(len(y_test), 784)
    print '\nFirst 5 labels of MNIST y_train: ', y_train[:5]
    y_train_ohe = np_utils.to_categorical(y_train)
    print '\nFirst 5 labels of MNIST y_train (one-hot):\n', y_train_ohe[:5]
    print ''
    return X_train, y_train, X_test, y_test, y_train_ohe

def define_nn_mlp_model(X_train, y_train_ohe):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    model = Sequential() # sequence of layers
    num_neurons_in_layer = 100 # number of neurons in a layer
    num_inputs = X_train.shape[1] # number of features (784)
    num_classes = y_train_ohe.shape[1]  # number of classes, 0-9
    model.add(Dense(input_dim=num_inputs,
                     units=num_neurons_in_layer*35,
                     kernel_initializer='uniform',
                     activation='tanh')) # only 12 neurons in this layer!
    model.add(Dense(input_dim=num_neurons_in_layer*35,
                     units=num_classes,
                     kernel_initializer='uniform',
                     activation='softmax')) # only 12 neurons - keep softmax at last layer
    sgd = SGD(lr=0.027, decay=1e-7, momentum=.91) # using stochastic gradient descent (keep)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) # (keep)
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict_classes(X_train, verbose=0)
    y_test_pred = model.predict_classes(X_test, verbose=0)
    print '\nRandom number generator seed: ', rng_seed
    print '\nFirst 30 labels:      ', y_train[:30]
    print 'First 30 predictions: ', y_train_pred[:30]
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print '\nTraining accuracy: %.2f%%' % (train_acc * 100)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print 'Test accuracy: %.2f%%' % (test_acc * 100)
    if test_acc < 0.95:
        print '\nMan, your test accuracy is bad! '
        print "Can't you get it up to 95%?"
    else:
        print "\nYou've made some improvements, I see..."


if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    print('rng_seed')
    X_train, y_train, X_test, y_test, y_train_ohe = load_and_condition_MNIST_data()
    print('split')
    np.random.seed(rng_seed)
    print('randomized')
    model = define_nn_mlp_model(X_train, y_train_ohe)
    print('model_defined')
    model.fit(X_train, y_train_ohe, epochs=40, batch_size=4000, verbose=1,
              validation_split=0.1) # cross val to estimate test error
    print('fit')
    print_output(model, y_train, y_test, rng_seed)
    print('output')
