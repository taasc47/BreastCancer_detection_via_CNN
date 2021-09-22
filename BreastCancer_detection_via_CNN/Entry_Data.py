# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:27:25 2020

@author: THISUM PC
"""


import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pickle

DATADIR = "C:/Users/THISUM PC/Desktop/breastCancer_ComputerVision & CNN/Breastimages"   # change this after coping
CATEGORIES = ["abnormal", "normal"]         #data set


IMG_SIZE = 227      # input images resize

training_data = []

def create_training_data():     #training data arranging
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):  # iterate over each image per abnormal and normal
                try:
                    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
            
            
            
create_training_data()  

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X= np.reshape(np.array(X), (len(X),IMG_SIZE,IMG_SIZE,1))
y= np.reshape(np.array(y),(len(y),1))

  
pickle_out = open("X.pickle","wb")  # input data saving
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")  # decided value for input data saving
pickle.dump(y, pickle_out)
pickle_out.close()