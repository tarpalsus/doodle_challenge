# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 16:24:03 2018

@author: kondrat
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from skimage.transform import resize
from skimage import data, io
from keras.preprocessing.sequence import pad_sequences
from ast import literal_eval

#data = pd.read_csv(r"C:\Users\kondrat\Documents\Python Scripts\train_simplified\calendar.csv")


POINT_COUNT = 100
STROKE_COUNT = 30

def drawing_to_np(drawing, shape=(100, 100), stroke_count=STROKE_COUNT):
    # evaluates the drawing array
    drawing = eval(drawing)
    #fig, ax = plt.subplots()
    np_drawing_channels = np.zeros((STROKE_COUNT, shape[0], shape[1]))
    # Close figure so it won't get displayed while transforming the set
    #plt.close(fig)
    for i, (x, y) in enumerate(drawing):
        if i < STROKE_COUNT:
            fig, ax = plt.subplots()
            ax.plot(x, y, marker='', color='black')
            ax.axis('off')  
            #fig = plt.figure()
            fig.canvas.draw()
            np_drawing = np.array(fig.canvas.renderer._renderer)
            np_drawing = np_drawing / 255.
            np_drawing = resize(np_drawing, shape, anti_aliasing=True)
            np_drawing =np_drawing[:, :, 1]    
            np_drawing_channels[i, :, :] = np_drawing
            ax.axis('off')        
            plt.close(fig)
        else:
            break

    # Normalize data
    #np_drawing = np_drawing.resize((shape[0],shape[1]), PIL.Image.ANTIALIAS)
    return np_drawing_channels


#out = drawing_to_np(drawing)
#io.imshow(out)
#plt.show()

def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    stroke_vec = literal_eval(raw_strokes) # string->list
    # unwrap the list
    in_strokes = [(xi,yi,i)  
     for i,(x,y) in enumerate(stroke_vec) 
     for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
   
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()

    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)
    
def stack_3d(raw_strokes):
    stroke_vec = literal_eval(raw_strokes)  
    in_strokes = [np.array(stroke) for stroke in stroke_vec]
    #max_stroke = max([stroke.shape[1] for stroke in in_strokes])
    padded = [np.pad(stroke, ((0,0),(0,POINT_COUNT-stroke.shape[1])), mode='constant') if POINT_COUNT-stroke.shape[1]>0
                              else stroke[:,:POINT_COUNT]for stroke in
                  in_strokes ]
#    
    c_strokes = np.stack(padded)
    #print(max_stroke)
    return pad_sequences(c_strokes.swapaxes(0, 1), 
                         maxlen=STROKE_COUNT, 
                         padding='post').swapaxes(0, 1)

def stack_3d_fast(drawings, stroke_count, point_count):
    X_all = np.zeros((len(drawings), stroke_count, 2 , point_count))
    for i, raw_strokes in enumerate(drawings):
        stroke_vec = literal_eval(raw_strokes)  
        padded = np.zeros((stroke_count, 2 , point_count))
        for j, stroke in enumerate(stroke_vec):
            if j < stroke_count:
                stroke = np.array(stroke)
                padded[j, :, :stroke.shape[1]] = stroke[:,:point_count]
        X_all[i,:,:,:] = padded
    return X_all


if __name__ == '__main__':

    data = pd.read_csv(r"C:\Users\user\Desktop\doodle\first_10.csv")
    drawings = data['drawing'][:1000]
#    drawing = drawings[16249]
#    x = drawing_to_np(drawing)
#    fig, ax = plt.subplots()
#    for i in range(2):
#        plt.imshow(x[i,:,:])
#    plt.show()
    x = _stack_it(drawing)    
    #x_3d, x_3d_padded = stack_3d(drawing)
    
    
    #max_stroke = max(stack_3d(drawing) for drawing in drawings)
    #y = data['word']
    #drawings = drawings[:100]
    #X = stack_3d_fast(drawings, STROKE_COUNT, POINT_COUNT)
    import time
    start = time.time()
    all_drawings = np.zeros((len(drawings), STROKE_COUNT, 100, 100))
    for i, drawing in enumerate(drawings):
        all_drawings[i, :, :, :] = drawing_to_np(drawing)
    
    
    
    print(time.time() - start)