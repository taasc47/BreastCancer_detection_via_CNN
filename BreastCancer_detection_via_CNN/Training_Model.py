# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 16:48:20 2020

@author: THISUM PC
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(96, (11, 11), strides=(4,4), padding = 'valid', input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2),padding = 'valid'))

model.add(Conv2D(256, (5, 5) , padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2),padding = 'valid'))

model.add(Conv2D(384, (3, 3) , padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(384, (3, 3) , padding = 'same'))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3) , padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2),padding = 'valid'))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

#model.add(Dense(64))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=16, epochs=15)

model.save('Trained_CNN.Model') #validation accuracy of 0.7143 achieved

model.summary()