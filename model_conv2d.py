# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:26:16 2018

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:14:31 2018

@author: kondrat
"""


#from ailab_util import AILabUtil
#if AILabUtil() != None:
#    print('run')

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, Conv2D, Dense, Dropout, Flatten, \
 MaxPooling2D, TimeDistributed, MaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
#from keras.layers import CuDNNLSTM as LSTM # this one is about 3x faster on GPU instances
from keras.metrics import top_k_categorical_accuracy
from test_util import stack_3d_fast, drawing_to_np
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mapk import mapk
import os
import keras

POINT_COUNT = 100
STROKE_COUNT = 30
NCATS = 340
DP_DIR = r"C:\Users\user\Desktop\doodle\c_shuffled"

def top_3_accuracy(x,y): 
    return top_k_categorical_accuracy(x,y, 3)
   

def model_time_dist(y):
    word_encoder = LabelEncoder()
    word_encoder.fit(y)
    
    
    model = Sequential()
    model.add(TimeDistributed(Conv2D(16, (3,3), data_format='channels_last'), input_shape=(STROKE_COUNT, POINT_COUNT, 1, 100, 100)))
    model.add(TimeDistributed(MaxPooling2D((3,3))))
    model.add(TimeDistributed(Conv2D(32, (3,3))))
    model.add(TimeDistributed(MaxPooling2D((3,3))))
    model.add(TimeDistributed(Conv2D(64, (3,3))))
    model.add(TimeDistributed(MaxPooling2D((3,3))))
    model.add(TimeDistributed(Flatten()))
    #model.add(Dropout(0.5))
    model.add(LSTM(128, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences = False))
    #model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(len(word_encoder.classes_), activation = 'softmax'))
    #optimizer = SGD(lr=0.01, decay=0, momentum=0.5, nesterov=True)
    optimizer = RMSprop(lr=0.001)
    model.compile(optimizer = optimizer, 
                              loss = 'categorical_crossentropy', 
                              metrics = ['categorical_accuracy', top_3_accuracy])
    model.summary()
    return model, word_encoder

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
    data = pd.read_csv(r"C:\Users\user\Desktop\doodle\all_10%.csv")
    x = list(data['word'].unique())
    words_mapping =  {k:v for v,k in enumerate(x)}
    #data = data[data['word'].isin(['airplane', 'ambulance', 'ant', 'arm', 'axe'])]
    drawings = data['drawing'][:50]
    all_drawings = np.zeros((len(drawings), STROKE_COUNT, 100, 100))
    for i, drawing in enumerate(drawings):
        all_drawings[i, :, :, :] = drawing_to_np(drawing,
                    stroke_count=STROKE_COUNT)
    #drawing = drawings[16249]   
    #x_3d, x_3d_padded = stack_3d(drawing)
    y = data['word'][:50]
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #with tf.device('/gpu:2'):
            model, word_encoder = model_time_dist(y)
            y = pd.get_dummies(y)
            X = all_drawings
            X = X[:, :, np.newaxis, :, :]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    	    random_state=2)
            history = model.fit(np.array(X_train), np.array(y_train), epochs=300,
                                validation_data=(np.array(X_test), np.array(y_test)), batch_size=128, verbose=2)		
            y_test_val = np.array(y_test)
            result = model.evaluate(X_test, y_test_val)
            y_test_val = np.argmax(y_test_val, axis=1)
            predicted = model.predict(X_test)
            predicted = np.argsort(predicted, axis=1)[:,-3:][:, ::-1]
            score = mapk(y_test_val.tolist(), predicted.tolist())
            print(score)