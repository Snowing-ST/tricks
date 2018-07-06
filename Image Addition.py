# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 08:45:33 2017

@author: situ.st.1
"""

import cv2
import numpy as np
import pandas as pd
#from __future__ import division
from matplotlib import pyplot as plt


img = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10D.jpg')
mask1 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10M.jpg')
#img1 = cv2.imread('/raid/home/situ/image_addition/5093F10D.jpg')
#mask1 = cv2.imread('/raid/home/situ/image_addition/5093F10M.jpg')

color = np.array([255,0,0]) #Blue
color = np.array([0,255,0]) #Green
color = np.array([0,0,255]) #Red
                
def imgAdd(img,mask1,color,x):
#color is an 1*3 array of coefficients
#x is transparent rate  
    mask = mask1.copy()
    mask[:,:,0] = mask[:,:,0]*color[0]
    mask[:,:,1] = mask[:,:,1]*color[1]
    mask[:,:,2] = mask[:,:,2]*color[2]
#    cv2.imshow('mask',mask)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()    
    img_mix = cv2.addWeighted(img,1,mask,x,0)
    return img_mix

mask1 = cv2.imread('C:/Users/situ.st.1/Desktop/SituXueying/7.2/5093F10M.jpg')  
color = np.array([0.690196078,0.870588235,0.937254902]) 
img_mix = imgAdd(img,mask1,color,0.5)

color1 = color[:]*0.5
img_mix1 = imgAdd(img,mask1,color1,1)            
cv2.imshow('img_mix', img_mix)
cv2.imshow('img_mix1', img_mix1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('C:/Users/situ.st.1/Desktop/SituXueying/7.2/img_mix.jpg',img_mix)
#cv2.imwrite("/raid/home/situ/image_addition/img_mix.jpg",img_mix)
cv2.imwrite('C:/Users/situ.st.1/Desktop/SituXueying/7.2/img_mix1.jpg',img_mix1)