# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:25:48 2020

@author: THISUM PC
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import timeit

start= timeit.default_timer()

def process(img):
    imgrgb = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    grayimg = cv2.cvtColor(imgrgb , cv2.COLOR_BGR2GRAY)
    
    picg = cv2.GaussianBlur(grayimg, (3, 3), 0)
    
    et, th = cv2.threshold(picg, 25,255, cv2.THRESH_BINARY)
    
    cannyPic = cv2.Canny(th, 100, 225)
    
    return cannyPic,imgrgb,grayimg

#=============================== Preprocess for the left breast =====================================================

def left(image,t):
    height, width = image.shape
    n=0
    lefty=np.zeros(image.shape)
    leftdot = []
    leftcood =[]
    for x in range(0,height,t):
        for y in range(width):
            if image[x,y]==255:
                n=n+1
                if n==1:
                    yref=y
                else:
                    grad=(y-yref)/t
                    yref=y
                    if grad<0:
                        lefty[x,y]=255
                        leftcood.append(x)
                        leftcood.append(y)
                        if len(leftdot)==0:
                            leftdot.append(grad)
                            leftdot.append(grad)
                            
                        elif len(leftdot)!=0:
                            leftdot.append(grad)
                            break
                    else:
                        leftdot.append(0)
                        break
            else:
                pass
    leftcood=np.array(leftcood)         #convert the list of coordinates to an array
    leftcood=np.reshape(leftcood,(int(len(leftcood)/2),2)) # left breast edge coordinates as a 2d array

    ycood=leftcood[:,1]
    yref=lefty.shape[1]
    pos=0

    for y in range(len(ycood)):     # loop to find the minimum y value of the breast edges
        if ycood[y]<yref:
            yref=ycood[y]
            pos=y

    ypos=yref 
    xpos=leftcood[pos,0]
    
    return xpos,ypos,lefty

#================================ Preprocess for the right breast ===================================================

def right(image,t):
    height, width = image.shape
    n=0
    righty=np.zeros(image.shape)
    rightdot = []
    rightcood =[]
    for x in range(0,height,t):
        for y in range(width):
            if image[x,y]==255:
                n=n+1
                if n==1:
                    yref=y
                else:
                    grad=(y-yref)/t
                    yref=y
                    if grad>0:          # consider positive peaks
                        righty[x,y]=255
                        rightcood.append(x)
                        rightcood.append(y)
                        if len(rightdot)==0:
                            rightdot.append(grad)
                            rightdot.append(grad)
                            
                        elif len(rightdot)!=0:
                            rightdot.append(grad)
                            break
                    else:
                        rightdot.append(0)
                        break
            else:
                pass
    rightcood=np.array(rightcood)         #convert the list of coordinates to an array
    rightcood=np.reshape(rightcood,(int(len(rightcood)/2),2)) # left breast edge coordinates as a 2d array

    ycood=rightcood[:,1]
    yref=0
    pos=0

    for y in range(len(ycood)):     # loop to find the minimum y value of the breast edges
        if ycood[y]>yref:
            yref=ycood[y]
            pos=y

    ypos=yref 
    xpos=rightcood[pos,0]
    
    return xpos,ypos,righty
    
#========================= GVF snake (active contour) segmentation ===================================================

def segmented(img):
    reduced=int(img.shape[0]/3)

    reducedPart=img[:reduced,:]
    adding=np.ones((reducedPart.shape),np.uint8)

    cannyLeft=process(img)[0][reduced:,:int(process(img)[0].shape[1]/2)]    #process(img)[0] is cannyPic
    cannyRight=process(img)[0][reduced:,int(process(img)[0].shape[1]/2):]

    #======================" Active contour for the left breast "==================================================#
    
    xpos=left(cannyLeft,4)[0]
    ypos=left(cannyLeft,4)[1]

    u=xpos*0.75                                     # x-position of the center
    b= ((left(cannyLeft,4)[2].shape[1]-ypos)/2)     # radius of the y-axis
    v=(ypos+b)*0.9                                  # y-position of the centre
    a=b*1.2                                         # radius on the x-axis


    t = np.linspace(0, 2*np.pi, 360)
    c = v+b*np.sin(t)
    r = u+a*np.cos(t)
    init1 = np.array([r, c]).T


    imgL=process(img)[1][reduced:,:int(img.shape[1]/2)]     #for the snake
    imgleft=img[reduced:,:int(img.shape[1]/2)]     #for the output

    snake1 = active_contour(imgL,
                       init1, alpha=0.6, beta=8, gamma=0.01,
                           w_line=0,w_edge=1, coordinates='rc',max_iterations=5000)


    imgPoly = np.zeros(cannyLeft.shape, np.uint8)

    snake1 = snake1.astype(np.int32)

    temp=np.copy(snake1[:, 0])
    snake1[:, 0]= snake1[:, 1]
    snake1[:, 1]=temp

    mask=cv2.fillPoly(imgPoly, [snake1], (255,255,255))
    Lmask = cv2.bitwise_and(imgleft,imgleft, mask=mask)

    #======================" Active contour for the Right breast "==================================================

    xpos=right(cannyRight,4)[0]
    ypos=right(cannyRight,4)[1]
    
    u=xpos*0.8                # x-position of the center
    #b=leftseg()[1]                    # radius of the y-axis
    v=(ypos-b)                 # y-position of the centre
    #a=leftseg()[2]                         # radius on the x-axis

    #t = np.linspace(0, 2*np.pi, 360)
    c = v+b*np.sin(t)
    r = u+a*np.cos(t)
    init2 = np.array([r, c]).T

    imgN=process(img)[1][reduced:,int(img.shape[1]/2):]     #for the snake
    imgright=img[reduced:,int(img.shape[1]/2):]     #for the output

    snake = active_contour(imgN,
                       init2, alpha=0.6, beta=0.6, gamma=0.01,
                           w_line=0,w_edge=1, coordinates='rc',max_iterations=5000)
   
    imgPoly = np.zeros(cannyRight.shape, np.uint8)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(imgN, cmap=plt.cm.gray)
    ax.plot(init2[:, 1], init2[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=4)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, imgN.shape[1], imgN.shape[0], 0])
    
    plt.show()"""


    snake = snake.astype(np.int32)
        
    temp=np.copy(snake[:, 0])
    snake[:, 0]= snake[:, 1]
    snake[:, 1]=temp

    mask=cv2.fillPoly(imgPoly, [snake], (255,255,255))
    Rmask = cv2.bitwise_and(imgright,imgright, mask=mask)
    
    hfit=np.hstack((Lmask,Rmask))

    final=np.vstack((adding,hfit))
    
    return final,snake1,snake,reduced

#img = cv2.imread('images/4.jpg')
#cv2.imshow('final',segmented(img)[0])
#segmented(img)


stop = timeit.default_timer()
print('Time for Preprocessing:',stop-start)

#cv2.waitKey(0)
#cv2.destroyAllWindows()