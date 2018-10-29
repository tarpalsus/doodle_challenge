# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:13:56 2018

@author: user
"""

import GPy, GPyOpt

import numpy as np

from mapk import mapk

POINT_COUNT = 100
STROKE_COUNT = 30

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, \
 MaxPooling2D
from sklearn.model_selection import train_test_split
#from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
from keras.metrics import top_k_categorical_accuracy
from test_util import stack_3d_fast
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping

from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def top_3_accuracy(x,y): 
    return top_k_categorical_accuracy(x,y, 3)

class DoodleModel():

    def __init__(self,  drawings, y, point_count=POINT_COUNT, 
                 stroke_count=STROKE_COUNT,
                 epochs=10, l1_drop=0.1,
                 l2_drop=0.1, l3_drop=0.1 ,l4_drop=0.1 ,l5_drop=0.1):
        X = stack_3d_fast(drawings, stroke_count, point_count)
        X = np.swapaxes(X, 2, 3)
        
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=2)
        self._y_test = np.array(self._y_test)
        self._y_train = np.array(self._y_train)
        self.epochs = epochs
        #self._x_train, self._y_train, self._x_test, self._y_test = x_train, y_train, x_test, y_test
        self.l1_drop = l1_drop
        self.l2_drop = l2_drop
        self.l3_drop = l3_drop
        self.l4_drop = l4_drop
        self.l5_drop = l5_drop
        self.point_count = point_count
        self.stroke_count = stroke_count
        
        self._model = self.stroke_model()
        self.batch_size = 128
        

    def stroke_model(self):

        #word_encoder = LabelEncoder()
        #word_encoder.fit(self._y_test)
        
        stroke_read_model = Sequential()
        #stroke_read_model.add(BatchNormalization(input_shape = (None,)+X.shape[2:]))
        # filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
        stroke_read_model.add(Conv2D(16, (3,3), input_shape=(self.stroke_count,
                                     self.point_count, 2 )))
        #stroke_read_model.add(Dropout(self.l1_drop)) #optimized to 0
        #stroke_read_model.add(MaxPooling2D(3,3))
        stroke_read_model.add(Conv2D(32, (5,5), padding='same'))
        stroke_read_model.add(Dropout(0.3)) #optimized to 0.3
        stroke_read_model.add(Conv2D(64, (5,5), padding='same'))
        
        stroke_read_model.add(Dropout(self.l1_drop))
        stroke_read_model.add(Conv2D(96, (3,3),padding='same'))
        stroke_read_model.add(MaxPooling2D(3,3,padding='same'))
        
        stroke_read_model.add(Conv2D(96, (3,3),padding='same'))
        stroke_read_model.add(MaxPooling2D(3,3,padding='same'))
        stroke_read_model.add(Flatten())
        
    #    stroke_read_model.add(LSTM(128, return_sequences = True))
        stroke_read_model.add(Dropout(self.l2_drop))
    #    stroke_read_model.add(LSTM(128, return_sequences = False))
        
        stroke_read_model.add(Dense(1024))
        stroke_read_model.add(Dropout(self.l3_drop))
        stroke_read_model.add(Dense(512))
        stroke_read_model.add(Dropout(self.l4_drop))
        stroke_read_model.add(Dense(128))
        stroke_read_model.add(Dropout(self.l5_drop))
        stroke_read_model.add(Dense(10, activation = 'softmax'))
        optimizer = SGD(lr=0.001, decay=0, momentum=0.5, nesterov=True)
        #optimizer = RMSprop(lr=0.001)
        stroke_read_model.compile(optimizer = optimizer, 
                                  loss = 'categorical_crossentropy', 
                                  metrics = ['categorical_accuracy', top_3_accuracy])#], top_3_accuracy])
        #stroke_read_model.summary()
        return stroke_read_model

    def model_fit(self):

        early_stopping = EarlyStopping(patience=0, verbose=1)
        self._model.fit(self._x_train, self._y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0,
                       validation_split=None,
                       callbacks=[early_stopping])
        
    def model_evaluate(self):
        self.model_fit()
        evaluation = self._model.evaluate(self._x_test, self._y_test,
                                           batch_size=self.batch_size, verbose=0)
        return evaluation


def run_model(drawings, y, stroke_count=STROKE_COUNT, point_count=POINT_COUNT,
              epochs=10, l1_drop=0.1,
              l2_drop=0.1, l3_drop=0.1, l4_drop=0.1 ,l5_drop=0.1):
    _model = DoodleModel(
                   drawings, y, stroke_count=stroke_count, point_count=point_count,
                   epochs=epochs, 
                   l1_drop=l1_drop, l2_drop=l2_drop, l3_drop=l3_drop, l4_drop=l4_drop,
                   l5_drop=l5_drop)
    
    model_evaluation = _model.model_evaluate()
    return model_evaluation


bounds = [
          {'name': 'l1_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
          {'name': 'l2_out',           'type': 'discrete',    'domain': (64, 128, 256, 512, 1024)},
          {'name': 'batch_size',       'type': 'discrete',    'domain': (10, 100, 500)},
          {'name': 'epochs',           'type': 'discrete',    'domain': (5, 10, 20)}]

bounds = [
          {'name': 'l1_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l2_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l3_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l4_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
          {'name': 'l5_drop',          'type': 'continuous',  'domain': (0.0, 0.3)},
        {'name': 'epochs',           'type': 'discrete',    'domain': (20, 50, 100)},
        {'name': 'stroke_count',           'type': 'discrete',    'domain': (20, 10, 30)},
        {'name': 'point_count',           'type': 'discrete',    'domain': (30, 50, 100)}]

def f(parameters, drawings, y):
    parameters = parameters[0]
    evaluation = run_model(
        drawings, y,
#        l1_drop=parameters[0],
#        l2_drop=parameters[1],
#        l3_drop=parameters[2],
#        l4_drop=parameters[3],
#        l5_drop=parameters[4],
#        epochs = int(parameters[5]),
        stroke_count=int(parameters[6]),
        point_count=int(parameters[7]),
        )
    print("LOSS:\t{0} \t ACCURACY:\t{1} \n TOP_3:\t{1}".format(evaluation[0], 
          evaluation[1], evaluation[2]))
    return evaluation[0]



data = pd.read_csv(r"first_10.csv")
#data = data[data['word'].isin(['airplane', 'ambulance', 'ant', 'arm', 'axe'])]
drawings = data['drawing']
#drawing = drawings[16249]   
#x_3d, x_3d_padded = stack_3d(drawing)
#max_stroke = max(stack_3d(drawing) for drawing in drawings)
y = data['word'][:10000]
y = pd.get_dummies(y)
drawings = drawings[:10000]

# optimizer
objective = partial(f, drawings=drawings, y=y)

opt_mnist = GPyOpt.methods.BayesianOptimization(f=objective, domain=bounds)

# optimize mnist model
opt_mnist.run_optimization(max_iter=10)


print("""

Optimized Parameters:

\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
\t{10}:\t{11}

""".format(bounds[0]["name"],opt_mnist.x_opt[0],
bounds[1]["name"],opt_mnist.x_opt[1],
bounds[2]["name"],opt_mnist.x_opt[2],
bounds[3]["name"],opt_mnist.x_opt[3],
bounds[6]["name"],opt_mnist.x_opt[6],
bounds[7]["name"],opt_mnist.x_opt[7]))

print("optimized loss: {0}".format(opt_mnist.fx_opt))