'''
Code implements multi-perceptron neural network to classify MNIST images of
handwritten digits using Keras and Theano.  Based on code from
https://www.packtpub.com/books/content/training-neural-networks-efficiently-using-keras

Note: neural network geometry not optimized (accuracy could be much better!)
'''

from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
import theano
import cv2
from sklearn import preprocessing

def load_and_condition_data():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
    X_train.resize(y_train.shape[0], (X_train.size//y_train.shape[0])) # 28 pix x 28 pix = 784 pixels
    X_test.resize(y_test.shape[0], (X_test.size//y_test.shape[0]))
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX) #before conversion were uint8
    X_test = X_test.astype(theano.config.floatX)
    print('\nFirst 5 labels of y_train: ', y_train[:5])
    return X_train, y_train, X_test, y_test

def define_nn_mlp_model(X_train, y_train):#_ohe):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too

    model = Sequential() # sequence of layers
    num_neurons_in_layer = 10 # number of neurons in a layer
    num_inputs = X_train.shape[1] # number of features (684288)
    model.add(Dense(input_dim=num_inputs,
                     units=num_neurons_in_layer,
                     activation='tanh')) # only 12 neurons in this layer!
    model.add(Dense(input_dim=num_neurons_in_layer,
                     units=1,
                     activation='linear')) # only 12 neurons - keep softmax at last layer
    sgd = SGD(lr=1.0, decay=1e-7, momentum=0.95) # using stochastic gradient descent (keep)
    model.compile(loss='msle', optimizer='adadelta', metrics=['msle'] ) # (keep)
    return model

def print_output(model, y_train, y_test, rng_seed):
    '''prints model accuracy results'''
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('\nTraining accuracy: %.2f%%' % (train_acc * 100))
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    if test_acc < 0.95:
        print('\nMan, your test accuracy is bad! ')
        print("Can't you get it up to 95%?")
    else:
        print("\nYou've made some improvements, I see...")


if __name__ == '__main__':
    rng_seed = 2 # set random number generator seed
    df = pd.read_csv('2016-17_advanced.csv')
    df = df[df['Yr'] == 2017]
    df = df[df['MP'] >= 1000]
    # df = df[df['Player_ID'] >= 0]
    # df = df[df['Player_ID'].duplicated()]
    # df = df[df['Pos'] != 'PG']
    # df = df[df['Pos'] != 'SG']
    df['Player_ID'] = df['Player_ID'].astype(int)
    y = np.array(df['OBPM'].values)
    pid = np.array(df['Player_ID'].values)
    images = []
    for i in range(len(pid)):
       images.append(cv2.imread('thumbnails/thumbnail_{0}_2016-17.png'.format(pid[i]), 0))
    X = np.array(images)
    flat_exes = []
    for i in range(len(X)):
        flat_exes.append(X[i].flatten())
    X = np.array(flat_exes)
    X_train, y_train, X_test, y_test = load_and_condition_data() #, y_train_ohe = load_and_condition_data()
    np.random.seed(rng_seed)
    model = define_nn_mlp_model(X_train, y_train) #ohe
    model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1,
              validation_split=0.2) # cross val to estimate test error #ohe
    # print_output(model, y_train, y_test, rng_seed)
