from __future__ import division
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import theano
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

if __name__ == "__main__":
    df = pd.read_csv('2016-17_advanced.csv')
    df = df[df['MP'] >= 1000]
    test = df[df['Player_ID'].duplicated()]
    test = test[test['Player_ID'] >= 0]
    df = df[df['Player_ID'].isin(test['Player_ID'])]
    df = df[df['Yr'] == 2017]
    # df = df[df['Player_ID'] >= 0]
    # df = df[df['Player_ID'].duplicated()]
    X = np.array(df['OBPM'].values)
    X = np.array(X).reshape(-1, 1)
    # pid = np.array(df['Player_ID'].values.astype('int'))
    # images = []
    # for i in range(len(pid)):
    #    images.append(cv2.imread('2016-17/{0}_2016.png'.format(pid[i]), 0).flatten())
    # X = np.array(images)
    # X = np.array(X)
    df2 = pd.read_csv('2016-17_advanced.csv')
    # df = df[df['Yr'] == 2017]
    df2 = df2[df2['MP'] >= 1000]
    df2 = df2[df2['Player_ID'] >= 0]
    df2 = df2[df2['Player_ID'].duplicated()]
    y = np.array(df2['OBPM'].values)
    y = np.array(y).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=21)
    test = []
    for i in np.arange(0.01, 1, 0.01):
        model = XGBRegressor(base_score=0.46229346552682926, learning_rate=i, n_estimators=50)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        test.append(score)
        print(i, score)
