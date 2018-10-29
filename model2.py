# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:14:31 2018

@author: kondrat
"""


from ailab_util import AILabUtil
if AILabUtil() != None:
    print('run')

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, \
 MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
from keras.metrics import top_k_categorical_accuracy
from test_util import stack_3d_fast
from keras.optimizers import SGD, Adam, RMSprop

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

POINT_COUNT = 100
STROKE_COUNT = 30

def top_3_accuracy(x,y): 
    return top_k_categorical_accuracy(x,y, 3)

def create_model(y):
    word_encoder = LabelEncoder()
    word_encoder.fit(y)
    
    
    stroke_read_model = Sequential()
    #stroke_read_model.add(BatchNormalization(input_shape = (None,)+X.shape[2:]))
    # filter count and length are taken from the script https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py
    stroke_read_model.add(Conv2D(16, (3,3), input_shape=(STROKE_COUNT, POINT_COUNT, 2 )))
    stroke_read_model.add(Dropout(0.3))
    #stroke_read_model.add(MaxPooling2D(3,3))
    stroke_read_model.add(Conv2D(32, (5,5)))
    stroke_read_model.add(Dropout(0.1))
    stroke_read_model.add(Conv2D(64, (5,5)))
    
    stroke_read_model.add(Dropout(0.1))
    stroke_read_model.add(Conv2D(96, (3,3)))
    stroke_read_model.add(MaxPooling2D(3,3))
    
    stroke_read_model.add(Conv2D(96, (3,3)))
    stroke_read_model.add(MaxPooling2D(3,3))
    stroke_read_model.add(Flatten())
    
#    stroke_read_model.add(LSTM(128, return_sequences = True))
    stroke_read_model.add(Dropout(0.3))
#    stroke_read_model.add(LSTM(128, return_sequences = False))
    
    stroke_read_model.add(Dense(1024))
    stroke_read_model.add(Dropout(0.1))
    stroke_read_model.add(Dense(512))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(128))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
    optimizer = SGD(lr=0.001, decay=0, momentum=0.5, nesterov=True)
    #optimizer = RMSprop(lr=0.001)
    stroke_read_model.compile(optimizer = optimizer, 
                              loss = 'categorical_crossentropy', 
                              metrics = ['categorical_accuracy', top_3_accuracy])
    stroke_read_model.summary()
    return stroke_read_model, word_encoder
    
def vis(history, save_path):
    'save training process plots'
    plt.plot(history['categorical_accuracy'])
    plt.plot(history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path+'_accuracy.png')
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_path+'_loss.png') 

if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\kondrat\.spyder-py3\first_10.csv")
    data = data[data['word'].isin(['airplane', 'ambulance', 'ant', 'arm', 'axe'])]
    drawings = data['drawing']
    #drawing = drawings[16249]   
    #x_3d, x_3d_padded = stack_3d(drawing)
    
    
    #max_stroke = max(stack_3d(drawing) for drawing in drawings)
    y = data['word']#[:100000]
    model, word_encoder = create_model(y)
    y = pd.get_dummies(y)
    #drawings = drawings[:100000]
    X = stack_3d_fast(drawings, STROKE_COUNT, POINT_COUNT)
    X = np.swapaxes(X, 2, 3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=2)
    history = model.fit(X_train, y_train, epochs=100,
                        validation_data=(X_test, y_test), batch_size=128)
    
    result = model.evaluate(X_test, y_test)
    