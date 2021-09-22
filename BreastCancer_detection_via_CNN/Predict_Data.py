
"""
Created on Sun Nov 15 18:08:23 2020

@author: THISUM PC
"""

import cv2
import tensorflow as tf
import timeit
import Preprocess as pre
import numpy as np

start= timeit.default_timer()


CATEGORIES = ["abnormal", "normal"]     # 0 for abnormal, 1 for normal

def prepare(image):
    IMG_SIZE = 227  # 227 input shape
    img_array = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


img = cv2.imread('images/4.jpg')
imgIn=pre.segmented(img)[0]


model = tf.keras.models.load_model("Trained_CNN.Model")

prediction = model.predict([prepare(imgIn)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])

#=================================== final Output display Process ===============================================#


imgup = img[:pre.segmented(img)[3],:,:]
imgleft=img[pre.segmented(img)[3]:,:int(img.shape[1]/2),:]
imgright=img[pre.segmented(img)[3]:,int(img.shape[1]/2):,:]

cv2.drawContours(imgleft,[pre.segmented(img)[1]], 0, (0,0,255), 1,cv2.LINE_4)
cv2.drawContours(imgright,[pre.segmented(img)[2]], 0, (0,0,255), 1,cv2.LINE_AA)

hfit=np.hstack((imgleft,imgright))
final=np.vstack((imgup,hfit))

#cv2.imshow('final',final)

scale_percent = 300 # percent of original size
#width = int(final.shape[1] * scale_percent / 100)
width = 750
#height = int(final.shape[0] * scale_percent / 100)
height = 600
dim = (width, height)
resized = cv2.resize(final, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)

status=CATEGORIES[int(prediction[0][0])]
resultingStatement = 'Result: '+status
font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 1
org = (10, 50)
fontColor = (255, 255, 255)
thikness = 2
cv2.putText(resized, resultingStatement, org, font, fontSize, fontColor, thikness, cv2.LINE_AA)

author='Breast Cancer via CNN by SACHINI FERNANDO'
cv2.putText(resized, author, (10,590), font, 0.7, fontColor, 2, cv2.LINE_AA)

cv2.imshow('PREDICTION', resized)
cv2.moveWindow('PREDICTION', 500, 100)

stop = timeit.default_timer()
print('Time:',stop-start)

cv2.imwrite('ab2.bmp',imgIn)
cv2.waitKey(0)
cv2.destroyAllWindows()

